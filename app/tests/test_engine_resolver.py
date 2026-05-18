import json

from app.core import engine_resolver


def test_hf_bundle_metadata_covers_all_extracted_engines(tmp_path, monkeypatch):
    engine_dir = tmp_path / "models" / "demo" / "engines"
    encoder = engine_dir / "encoder.plan"
    decoder = engine_dir / "decoder.plan"

    def fake_fetch_manifest(model_id):
        assert model_id == "demo"
        return {
            "files": {
                "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz": {
                    "sha256": "unused",
                    "size": 123,
                }
            }
        }

    def fake_download_and_extract_tarball(rel_path, dest_dir, expected_sha256=None):
        assert rel_path == "models/demo/engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz"
        assert expected_sha256 == "unused"
        dest_dir.mkdir(parents=True, exist_ok=True)
        encoder.write_bytes(b"encoder")
        decoder.write_bytes(b"decoder")
        (dest_dir / "._decoder.plan").write_bytes(b"macos metadata")

    from app.core import hf_artifacts

    monkeypatch.setattr(hf_artifacts, "fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(
        hf_artifacts,
        "download_and_extract_tarball",
        fake_download_and_extract_tarball,
    )

    spec = engine_resolver.EngineSpec(
        model_id="demo",
        engine_file="encoder.plan",
        engine_path=encoder,
        env_var="ENC_ENGINE",
        onnx_input=None,
        build_script=None,
        build_env={},
        hf_only=True,
        required=True,
    )
    host = engine_resolver.HostSignature("87", "10.3", "6.2", "12.6")

    assert engine_resolver._try_hf_resolve(spec, host)

    encoder_meta = json.loads((engine_dir / "encoder.plan.meta.json").read_text())
    decoder_meta = json.loads((engine_dir / "decoder.plan.meta.json").read_text())
    assert encoder_meta["source"] == "hf_bundle"
    assert decoder_meta["source"] == "hf_bundle"
    assert encoder_meta["host"] == host.to_dict()
    assert decoder_meta["host"] == host.to_dict()
    assert not (engine_dir / "._decoder.plan.meta.json").exists()
