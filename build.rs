fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the gRPC schema. We keep the generated code under
    // `OUT_DIR` and pull it in via tonic::include_proto! at use
    // sites. tonic 0.12 ships its own protoc shim — no system
    // protoc needed.
    println!("cargo:rerun-if-changed=proto/velox.proto");
    tonic_build::configure()
        .build_client(false)
        .build_server(true)
        .compile(&["proto/velox.proto"], &["proto"])?;
    Ok(())
}
