pub mod db;
pub mod embedder;
pub mod index;
pub mod search;
pub mod graph;
pub mod chunker;
pub mod rrf;
pub mod reranker;
pub mod pipeline;
pub mod error;
pub mod frontmatter;
pub mod variants;
pub mod analyzers;
pub mod download;

#[cfg(test)]
mod local_embeddings_contract_tests {
    use tempfile::TempDir;

    use crate::download::{download_local_embedding_model, LocalEmbeddingModel};

    #[test]
    fn default_path_constructs_no_network_request_and_creates_no_cache_dir() {
        let cache = TempDir::new().unwrap().path().join("models");
        let embeddings = LocalEmbeddingModel::disabled(cache.clone());
        let request_constructions = 0usize;

        assert!(!embeddings.is_enabled());
        assert!(!embeddings.is_available());
        assert!(!cache.exists());
        assert_eq!(request_constructions, 0);
        // Network oracle: this local-only type exposes no client/request; its
        // separate explicit download entry point is the guarded stub below.
    }

    #[test]
    fn enabling_is_explicit_and_does_not_download_or_create_cache() {
        let cache = TempDir::new().unwrap().path().join("models");
        let embeddings = LocalEmbeddingModel::enabled(cache.clone());

        assert!(embeddings.is_enabled());
        assert!(!embeddings.is_available());
        assert!(!cache.exists());
    }

    #[test]
    fn explicit_download_step_is_guarded_without_network_or_filesystem_side_effects() {
        let cache = TempDir::new().unwrap().path().join("models");

        let error = download_local_embedding_model(&cache).unwrap_err();

        assert!(error.to_string().contains("automatic model download is disabled"));
        assert!(!cache.exists());
    }

    #[test]
    #[ignore = "requires an explicitly provisioned local ONNX model"]
    fn explicitly_provisioned_model_becomes_available() {
        let model_dir = std::env::var_os("VELOCIRAG_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set VELOCIRAG_TEST_MODEL_DIR");

        assert!(LocalEmbeddingModel::enabled(model_dir).is_available());
    }
}

#[cfg(test)]
mod offline_download_tests {
    use tempfile::TempDir;

    use crate::download::{
        ensure_model, ensure_model_in, is_offline, model_files_in,
        models_cache_dir, EMBEDDER_MODEL, FETCH_COMMAND, RERANKER_MODEL,
    };

    // Offline is the default and the flag is process-wide: no test may call
    // `set_offline(false)`, or a test running beside it could download.

    #[test]
    fn ensure_model_offline_creates_nothing_under_a_temp_home() {
        assert!(is_offline(), "offline must be the default");
        let home = TempDir::new().unwrap();
        let models = home.path().join(".cache").join("velocirag").join("models");

        for spec in [&EMBEDDER_MODEL, &RERANKER_MODEL] {
            let error = ensure_model_in(&models, spec).unwrap_err().to_string();
            let folder = models.join(spec.local_dir).display().to_string();
            assert!(error.contains(&folder), "the error must name the folder: {error}");
            assert!(error.contains(FETCH_COMMAND), "the error must name the command: {error}");
        }
        // The whole text, once: the loader gives the same one.
        let folder = models.join(EMBEDDER_MODEL.local_dir);
        assert_eq!(
            ensure_model_in(&models, &EMBEDDER_MODEL).unwrap_err().to_string(),
            format!(
                "Embedding error: the model files are not in {}. Copy model.onnx (or onnx/model.onnx) \
                 and tokenizer.json into it, or run \"axel-v2 model fetch\".",
                folder.display()
            )
        );
        let loader = match crate::embedder::Embedder::from_model_dir(&folder, 1, true) {
            Ok(_) => panic!("a folder that does not exist is not a model"),
            Err(error) => error.to_string(),
        };
        assert_eq!(loader, ensure_model_in(&models, &EMBEDDER_MODEL).unwrap_err().to_string());

        assert_eq!(
            std::fs::read_dir(home.path()).unwrap().count(),
            0,
            "nothing may be created under HOME"
        );
    }

    #[test]
    fn ensure_model_offline_returns_a_folder_that_has_the_files() {
        let home = TempDir::new().unwrap();
        let models = home.path().join("models");
        let folder = models.join(EMBEDDER_MODEL.local_dir);
        std::fs::create_dir_all(folder.join("onnx")).unwrap();
        std::fs::write(folder.join("onnx").join("model.onnx"), b"").unwrap();
        std::fs::write(folder.join("tokenizer.json"), b"").unwrap();

        assert_eq!(ensure_model_in(&models, &EMBEDDER_MODEL).unwrap(), folder);
    }

    #[test]
    fn ensure_model_never_creates_the_models_folder_of_this_home() {
        // The entry point itself, under whatever HOME the suite runs with
        // (an empty one under scripts/dirty-home.sh).
        let folder = models_cache_dir().join(EMBEDDER_MODEL.local_dir);
        let existed = folder.exists();

        match ensure_model(&EMBEDDER_MODEL) {
            Ok(found) => {
                assert!(existed, "a folder that was not there cannot be returned offline");
                assert_eq!(found, folder);
            }
            Err(error) => {
                let error = error.to_string();
                assert!(error.contains(&folder.display().to_string()), "{error}");
                assert!(error.contains(FETCH_COMMAND), "{error}");
            }
        }

        assert_eq!(folder.exists(), existed);
    }

    #[test]
    fn the_fetch_command_is_the_installed_name() {
        assert_eq!(FETCH_COMMAND, "axel-v2 model fetch");
    }

    // The files are empty stand-ins: these functions only check existence.

    /// A models folder holding the embedding model's folder with `files`.
    fn models_folder_with(files: &[&str]) -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
        let home = TempDir::new().unwrap();
        let models = home.path().join("models");
        let folder = models.join(EMBEDDER_MODEL.local_dir);
        for file in files {
            let path = folder.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"").unwrap();
        }
        (home, models, folder)
    }

    #[test]
    fn the_flat_layout_alone_passes_the_offline_check() {
        assert!(is_offline());
        let (_home, models, folder) = models_folder_with(&["model.onnx", "tokenizer.json"]);

        assert_eq!(model_files_in(&folder), Some(folder.join("model.onnx")));
        assert_eq!(ensure_model_in(&models, &EMBEDDER_MODEL).unwrap(), folder);
        assert!(!folder.join("onnx").exists(), "nothing may be created");
    }

    #[test]
    fn the_hub_layout_passes_the_offline_check_and_is_preferred() {
        let (_home, models, folder) = models_folder_with(&["onnx/model.onnx", "tokenizer.json"]);
        assert_eq!(model_files_in(&folder), Some(folder.join("onnx").join("model.onnx")));
        assert_eq!(ensure_model_in(&models, &EMBEDDER_MODEL).unwrap(), folder);

        let (_home, models, folder) =
            models_folder_with(&["model.onnx", "onnx/model.onnx", "tokenizer.json"]);
        assert_eq!(model_files_in(&folder), Some(folder.join("onnx").join("model.onnx")));
        assert_eq!(ensure_model_in(&models, &EMBEDDER_MODEL).unwrap(), folder);
    }

    #[test]
    fn a_graph_without_the_tokenizer_or_a_tokenizer_without_a_graph_is_not_a_model() {
        for files in [&["model.onnx"][..], &["onnx/model.onnx"][..], &["tokenizer.json"][..]] {
            let (_home, models, folder) = models_folder_with(files);
            assert_eq!(model_files_in(&folder), None, "{files:?}");
            let error = ensure_model_in(&models, &EMBEDDER_MODEL).unwrap_err().to_string();
            assert!(error.contains(&folder.display().to_string()), "{error}");
            assert!(error.contains(FETCH_COMMAND), "{error}");
        }
    }

    #[test]
    fn the_loader_and_the_offline_check_agree_on_both_layouts() {
        // The loader gets past its own file check exactly when the offline
        // check accepts the folder: with stand-in files it then fails in
        // ONNX Runtime, not with the "files are not in" message.
        for files in [&["model.onnx", "tokenizer.json"][..], &["onnx/model.onnx", "tokenizer.json"][..]] {
            let (_home, models, folder) = models_folder_with(files);
            assert!(ensure_model_in(&models, &EMBEDDER_MODEL).is_ok());
            let error = match crate::embedder::Embedder::from_model_dir(&folder, 1, true) {
                Ok(_) => panic!("an empty file is not a model"),
                Err(error) => error.to_string(),
            };
            assert!(!error.contains("are not in"), "{files:?}: {error}");
            assert!(error.contains("Failed to load ONNX model"), "{files:?}: {error}");
        }
        let (_home, models, folder) = models_folder_with(&["model.onnx"]);
        assert!(ensure_model_in(&models, &EMBEDDER_MODEL).is_err());
        let error = match crate::embedder::Embedder::from_model_dir(&folder, 1, true) {
            Ok(_) => panic!("a folder without a tokenizer is not a model"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("are not in") && error.contains(FETCH_COMMAND), "{error}");
    }
}

/// Embedding dimension for the default model (all-MiniLM-L6-v2).
/// Single source of truth — used by db.rs, embedder.rs, and tests.
pub const EMBEDDING_DIM: usize = 384;
