//! Write compact candidate fixtures into a new directory; never bless in tests.
#![forbid(unsafe_code)]

use linear_srgb_goldens as goldens;

fn main() {
    let directory = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .expect("usage: numerical_goldens <new-output-directory>"),
    );
    std::fs::create_dir(&directory).expect("capture directory must not already exist");
    let directory = directory.join(goldens::profile());
    std::fs::create_dir(&directory).unwrap();
    for (name, contents) in goldens::capture() {
        assert!(
            contents.len() <= 30_000,
            "split the generated {name} fixture before committing"
        );
        let path = directory.join(name);
        std::fs::write(&path, contents).unwrap();
        println!("{}", path.display());
    }
}
