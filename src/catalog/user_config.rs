// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The user-override layer of the Catalog: a TOML file the caller supplies.
//!
//! This is the highest-precedence layer (user > fetched > baseline). It is
//! parsed into per-Provider blocks, each carrying Provider-wide defaults plus a
//! list of Models; every Model is merged over the baseline entry with the same
//! Provider and id, or added when none exists, with the user value winning.
//!
//! The file mirrors tapir's own config idiom (TOML, like the Token Store file).
//! [`default_path`] resolves its conventional location via the XDG base
//! directories; the caller passes the result — or any path of its own — to
//! [`ModelRegistry::load`](super::ModelRegistry::load), the same way it hands an
//! explicit path to the file-backed Token Store.
//!
//! ```toml
//! [providers.openai]
//! # Provider-wide defaults, inherited by every Model in this block.
//! base_url = "https://gateway.internal"
//! auth_header = true
//!
//! # Override a baseline Model: only the given fields change.
//! [[providers.openai.models]]
//! id = "gpt-4o-mini"
//! max_tokens = 8192
//!
//! # Add a Model the baseline does not carry: the required fields must be given.
//! [[providers.openai.models]]
//! id = "gpt-internal"
//! name = "Internal GPT"
//! api = "openai-completions"
//! context_window = 128000
//! max_tokens = 16384
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::error::{Error, ErrorKind};
use crate::model::{
    Api, CompatConfig, InputType, Model, ModelCost, ModelEntry, ModelId,
    ProviderId,
};
use crate::registry::Registry;

/// The conventional path of the user-override file, via the XDG base
/// directories: `$XDG_CONFIG_HOME/tapir/models.toml`, falling back to
/// `$HOME/.config/tapir/models.toml`.
///
/// Returns `None` when neither variable is set, so a caller can pass the result
/// straight to [`ModelRegistry::load`](super::ModelRegistry::load), which treats
/// a `None` path — or a path with no file — as "no user layer".
#[must_use]
pub fn default_path() -> Option<PathBuf> {
    if let Some(dir) =
        std::env::var_os("XDG_CONFIG_HOME").filter(|d| !d.is_empty())
    {
        return Some(PathBuf::from(dir).join("tapir").join("models.toml"));
    }
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("tapir")
            .join("models.toml"),
    )
}

/// The whole user-override file: Provider id → its override block.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UserConfig {
    #[serde(default)]
    providers: BTreeMap<String, ProviderBlock>,
}

/// One Provider's overrides: Provider-wide defaults plus a list of Models.
///
/// The default fields (`base_url`..`compat`) are inherited by every Model in
/// `models` unless that Model overrides them; a Model's own field always wins.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderBlock {
    base_url: Option<String>,
    api: Option<Api>,
    api_key: Option<String>,
    auth_header: Option<bool>,
    headers: Option<Vec<(String, String)>>,
    compat: Option<CompatConfig>,
    #[serde(default)]
    models: Vec<UserModel>,
}

/// One Model's overrides: the Model's own description. Connection fields
/// (`base_url`, `api_key`, `auth_header`, `headers`) are Provider-wide and live
/// on [`ProviderBlock`], not here. Every field but `id` is optional: on an
/// existing baseline Model an absent field is left untouched; on a new Model the
/// required fields (`api`, `name`, `context_window`, `max_tokens`, and a
/// `base_url` from the Provider block) must be present.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UserModel {
    id: String,
    name: Option<String>,
    api: Option<Api>,
    reasoning: Option<bool>,
    input: Option<Vec<InputType>>,
    cost: Option<ModelCost>,
    context_window: Option<u32>,
    max_tokens: Option<u32>,
    compat: Option<CompatConfig>,
}

/// Parse `raw` and merge its overrides into `entries`, user value winning.
///
/// Each Provider key is canonicalized against the [`Registry`]; an alias (a name
/// that selects a Provider but is not its canonical id) is rejected, as is a
/// second key that resolves to a Provider already seen. Every Model in a block
/// is then merged over the baseline entry with the same Provider and id, or
/// pushed as a new entry when none matches.
///
/// # Errors
///
/// [`Decode`](ErrorKind::Decode) when the TOML is malformed;
/// [`InvalidRequest`](ErrorKind::InvalidRequest) for an aliased or duplicate
/// Provider key, an empty id, or a new Model missing a required field.
pub(super) fn apply(
    entries: &mut Vec<ModelEntry>,
    raw: &str,
) -> Result<(), Error> {
    let config: UserConfig = toml::from_str(raw).map_err(Error::decode)?;
    let mut seen: Vec<String> = Vec::new();
    for (key, block) in &config.providers {
        let canonical = canonical_provider(key)?;
        if seen.iter().any(|c| c.eq_ignore_ascii_case(&canonical)) {
            return Err(duplicate_provider(&canonical));
        }
        seen.push(canonical.clone());
        let provider_id = ProviderId::new(canonical)?;
        for user_model in &block.models {
            apply_model(entries, &provider_id, block, user_model)?;
        }
    }
    Ok(())
}

/// Canonicalize a Provider key, rejecting an alias.
///
/// A key that selects a compiled-in Provider must be that Provider's canonical
/// id (case aside); an alias is rejected so a duplicate can never slip in under
/// a second name. A key that selects no compiled-in Provider is taken verbatim.
fn canonical_provider(key: &str) -> Result<String, Error> {
    match Registry::resolve(key) {
        Some(info) if key.eq_ignore_ascii_case(info.id.as_str()) => {
            Ok(info.id.as_str().to_owned())
        }
        Some(info) => Err(aliased_provider(key, info.id.as_str())),
        None => Ok(key.to_owned()),
    }
}

/// Merge one Model's overrides into `entries`: patch the matching baseline entry
/// or push a new one.
fn apply_model(
    entries: &mut Vec<ModelEntry>,
    provider_id: &ProviderId,
    block: &ProviderBlock,
    user_model: &UserModel,
) -> Result<(), Error> {
    let id = ModelId::new(user_model.id.clone())?;
    match entries.iter_mut().find(|entry| {
        entry.model.provider == *provider_id && entry.model.id == id
    }) {
        Some(entry) => patch_entry(entry, block, user_model),
        None => entries.push(build_entry(provider_id, &id, block, user_model)?),
    }
    Ok(())
}

/// Apply Provider-wide defaults then the Model's own overrides to an existing
/// entry; the Model's value wins over the Provider default, which wins over the
/// baseline.
fn patch_entry(
    entry: &mut ModelEntry,
    block: &ProviderBlock,
    user_model: &UserModel,
) {
    // Provider-wide connection defaults apply to the whole block; the Model
    // overrides its own description on top of the baseline.
    let model = &mut entry.model;
    if let Some(v) = &block.base_url {
        model.base_url = v.clone();
    }
    if let Some(v) = &block.headers {
        model.headers = v.clone();
    }
    if let Some(v) = &block.api {
        model.api = v.clone();
    }
    if let Some(v) = &user_model.api {
        model.api = v.clone();
    }
    if let Some(v) = &user_model.name {
        model.name = v.clone();
    }
    if let Some(v) = user_model.reasoning {
        model.reasoning = v;
    }
    if let Some(v) = &user_model.input {
        model.input = v.clone();
    }
    if let Some(v) = user_model.cost {
        model.cost = v;
    }
    if let Some(v) = user_model.context_window {
        model.context_window = v;
    }
    if let Some(v) = user_model.max_tokens {
        model.max_tokens = v;
    }
    if let Some(v) = block.api_key.as_ref() {
        entry.api_key = Some(v.clone());
    }
    if let Some(v) = block.auth_header {
        entry.auth_header = v;
    }
    entry.compat = merge_compat(
        entry.compat.take(),
        block.compat.as_ref(),
        user_model.compat.as_ref(),
    );
}

/// Build a brand-new entry for a Model the baseline does not carry.
///
/// A [`Model`]'s description (`api`, `name`, the `context_window`/`max_tokens`
/// limits) comes from the Model, its connection (`base_url`, `headers`, and the
/// entry's Credential fields) from the Provider block; a missing required field
/// is an error. The purely optional fields default (`reasoning` off, text-only
/// input, zero cost, no headers).
fn build_entry(
    provider_id: &ProviderId,
    id: &ModelId,
    block: &ProviderBlock,
    user_model: &UserModel,
) -> Result<ModelEntry, Error> {
    let api = user_model
        .api
        .clone()
        .or_else(|| block.api.clone())
        .ok_or_else(|| missing_field(id, "api"))?;
    let name = user_model
        .name
        .clone()
        .ok_or_else(|| missing_field(id, "name"))?;
    // Connection fields come from the Provider block, so a new Model needs one
    // that sets base_url.
    let base_url = block
        .base_url
        .clone()
        .ok_or_else(|| missing_field(id, "base_url"))?;
    let context_window = user_model
        .context_window
        .ok_or_else(|| missing_field(id, "context_window"))?;
    let max_tokens = user_model
        .max_tokens
        .ok_or_else(|| missing_field(id, "max_tokens"))?;

    let model = Model {
        id: id.clone(),
        provider: provider_id.clone(),
        api,
        name,
        base_url,
        reasoning: user_model.reasoning.unwrap_or(false),
        input: user_model
            .input
            .clone()
            .unwrap_or_else(|| vec![InputType::Text]),
        cost: user_model.cost.unwrap_or_default(),
        context_window,
        max_tokens,
        headers: block.headers.clone().unwrap_or_default(),
    };

    let mut entry = ModelEntry::new(model);
    entry.api_key = block.api_key.clone();
    entry.auth_header = block.auth_header.unwrap_or(false);
    entry.compat =
        merge_compat(None, block.compat.as_ref(), user_model.compat.as_ref());
    Ok(entry)
}

/// Layer `provider` then `model` Compat overrides over `base`, field by field.
///
/// Each `CompatConfig` field is itself optional, so an override sets only the
/// fields it names and leaves the rest of `base` intact. Returns `None` when
/// nothing at any layer touches Compat, so an untouched entry keeps its
/// baseline value (which may itself be `None`).
fn merge_compat(
    base: Option<CompatConfig>,
    provider: Option<&CompatConfig>,
    model: Option<&CompatConfig>,
) -> Option<CompatConfig> {
    if base.is_none() && provider.is_none() && model.is_none() {
        return None;
    }
    let mut merged = base.unwrap_or_default();
    if let Some(src) = provider {
        overlay_compat(&mut merged, src);
    }
    if let Some(src) = model {
        overlay_compat(&mut merged, src);
    }
    Some(merged)
}

/// Copy every set field of `src` onto `dst`, leaving `dst`'s unset-in-`src`
/// fields untouched.
fn overlay_compat(dst: &mut CompatConfig, src: &CompatConfig) {
    if src.streaming.is_some() {
        dst.streaming = src.streaming;
    }
    if src.tools.is_some() {
        dst.tools = src.tools;
    }
    if src.multimodal.is_some() {
        dst.multimodal = src.multimodal;
    }
    if src.dialect.is_some() {
        dst.dialect = src.dialect;
    }
    if src.thinking.is_some() {
        dst.thinking = src.thinking;
    }
    if src.thinking_default.is_some() {
        dst.thinking_default = src.thinking_default;
    }
    if src.thinking_level_map.is_some() {
        dst.thinking_level_map = src.thinking_level_map.clone();
    }
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error for a Provider key
/// given as an alias rather than its canonical id.
fn aliased_provider(key: &str, canonical: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!(
            "user config provider {key:?} is an alias for {canonical:?}; use the canonical id"
        ),
    )
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error for two Provider keys
/// that resolve to the same Provider.
fn duplicate_provider(canonical: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!(
            "user config has more than one block for provider {canonical:?}"
        ),
    )
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error for a new Model that
/// omits a field a [`Model`] cannot be built without.
fn missing_field(id: &ModelId, field: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!(
            "user config adds Model {:?} but is missing required field {field:?}",
            id.as_str()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn openai_model(id: &str) -> Model {
        Model {
            id: ModelId::new(id).unwrap(),
            provider: ProviderId::from_static("openai"),
            api: Api::OpenAICompletions,
            name: "seed".to_owned(),
            base_url: "https://api.openai.com".to_owned(),
            reasoning: false,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 8_000,
            max_tokens: 4_000,
            headers: Vec::new(),
        }
    }

    fn baseline() -> Vec<ModelEntry> {
        vec![ModelEntry::new(openai_model("gpt-4o-mini"))]
    }

    #[test]
    fn override_patches_only_the_named_fields() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [[providers.openai.models]]
            id = "gpt-4o-mini"
            max_tokens = 8192
            "#,
        )
        .unwrap();
        let entry = &entries[0];
        assert_eq!(entry.model.max_tokens, 8192);
        // Untouched fields keep their baseline value.
        assert_eq!(entry.model.name, "seed");
        assert_eq!(entry.model.context_window, 8_000);
    }

    #[test]
    fn a_new_model_is_added() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [providers.openai]
            base_url = "https://gw.internal"

            [[providers.openai.models]]
            id = "gpt-internal"
            name = "Internal GPT"
            api = "openai-completions"
            context_window = 128000
            max_tokens = 16384
            "#,
        )
        .unwrap();
        assert_eq!(entries.len(), 2);
        let added = entries
            .iter()
            .find(|e| e.model.id.as_str() == "gpt-internal")
            .unwrap();
        assert_eq!(added.model.name, "Internal GPT");
        assert_eq!(added.model.base_url, "https://gw.internal");
        // Optional fields fall back to their defaults.
        assert_eq!(added.model.input, vec![InputType::Text]);
        assert!(!added.model.reasoning);
    }

    #[test]
    fn provider_defaults_are_inherited_by_every_model() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [providers.openai]
            base_url = "https://provider.default"
            auth_header = true

            [[providers.openai.models]]
            id = "gpt-4o-mini"

            [[providers.openai.models]]
            id = "gpt-2"
            name = "Two"
            api = "openai-completions"
            context_window = 1000
            max_tokens = 500
            "#,
        )
        .unwrap();
        // The Provider-wide connection defaults reach the patched baseline
        // Model and the newly added one alike.
        let patched = entries
            .iter()
            .find(|e| e.model.id.as_str() == "gpt-4o-mini")
            .unwrap();
        assert_eq!(patched.model.base_url, "https://provider.default");
        assert!(patched.auth_header);
        let added = entries
            .iter()
            .find(|e| e.model.id.as_str() == "gpt-2")
            .unwrap();
        assert_eq!(added.model.base_url, "https://provider.default");
        assert!(added.auth_header);
    }

    #[test]
    fn api_key_from_config_surfaces_on_the_entry() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [providers.openai]
            api_key = "sk-from-config"

            [[providers.openai.models]]
            id = "gpt-4o-mini"
            "#,
        )
        .unwrap();
        assert_eq!(entries[0].api_key.as_deref(), Some("sk-from-config"));
    }

    #[test]
    fn provider_headers_are_applied() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [providers.openai]
            headers = [["X-Org", "acme"]]

            [[providers.openai.models]]
            id = "gpt-4o-mini"
            "#,
        )
        .unwrap();
        assert_eq!(
            entries[0].model.headers,
            vec![("X-Org".to_owned(), "acme".to_owned())]
        );
    }

    #[test]
    fn compat_overrides_merge_field_by_field() {
        let mut entries = baseline();
        apply(
            &mut entries,
            r#"
            [providers.openai.compat]
            streaming = true

            [[providers.openai.models]]
            id = "gpt-4o-mini"
            compat = { tools = false }
            "#,
        )
        .unwrap();
        let compat = entries[0].compat.as_ref().unwrap();
        assert_eq!(compat.streaming, Some(true));
        assert_eq!(compat.tools, Some(false));
    }

    // Alias and duplicate detection lean on the compiled-in Registry, so they
    // only have Providers to resolve against when a Provider feature is on.
    #[cfg(feature = "openai")]
    #[test]
    fn an_alias_provider_key_is_rejected() {
        let mut entries = baseline();
        let err = apply(
            &mut entries,
            r#"
            [[providers.gpt.models]]
            id = "gpt-4o-mini"
            "#,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
    }

    #[cfg(feature = "openai")]
    #[test]
    fn two_keys_for_one_provider_are_rejected() {
        let mut entries = baseline();
        let err = apply(
            &mut entries,
            r#"
            [providers.openai]
            [providers.OpenAI]
            "#,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
    }

    #[test]
    fn a_new_model_missing_a_required_field_is_rejected() {
        let mut entries = baseline();
        let err = apply(
            &mut entries,
            r#"
            [[providers.openai.models]]
            id = "gpt-incomplete"
            name = "Incomplete"
            "#,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
        assert!(err.message().contains("api"));
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let mut entries = baseline();
        let err = apply(
            &mut entries,
            r#"
            [[providers.openai.models]]
            id = "gpt-4o-mini"
            nonsense = true
            "#,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }

    #[test]
    fn malformed_toml_is_a_decode_error() {
        let mut entries = baseline();
        let err = apply(&mut entries, "this is not = = toml").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }
}
