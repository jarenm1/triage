use std::{
    collections::HashSet,
    ffi::OsString,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use sim_inspection::Inspector;
use sim_scene::{GeneratorRecipe, SemanticClass};

const HELP: &str = "render-smoke --generate [--recipe FILE | --seed U32 --width U32 --height U32]
    [--start-index U32] [--count U32] --output DIRECTORY
render-smoke --generate --scene-only [same recipe options] [--start-index U32]
    [--output FILE]

Generate synchronized captures with one shared seeded scene generator.
Recipe, seed and dimensions default to GeneratorRecipe::default(); start-index=0, count=1.
--recipe reads a complete recipe JSON file and cannot be combined with --seed,
--width or --height. Duplicate flags and unknown arguments are rejected.
--scene-only requires count=1, emits the exact realized scene JSON without a GPU,
and writes stdout unless --output FILE is supplied. Output paths must not exist.
Batches publish only after every capture and manifest is complete. Each sample's
scene.json can be replayed with --inspect --scene PATH --output NEW_DIRECTORY.";

#[derive(Serialize)]
struct Sample {
    sample_index: u32,
    directory: String,
    scene: String,
    metadata: String,
}

pub async fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(2);
    let mut seen = HashSet::new();
    let mut recipe_path = None;
    let mut seed = None;
    let mut width = None;
    let mut height = None;
    let mut start_index = 0;
    let mut count = 1;
    let mut output = None;
    let mut scene_only = false;
    while let Some(arg) = args.next() {
        let flag = arg.to_str().context("generation flags must be UTF-8")?;
        ensure!(
            seen.insert(flag.to_owned()),
            "duplicate generation argument: {flag}"
        );
        match flag {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--seed" => seed = Some(number(&mut args, flag)?),
            "--width" => width = Some(number(&mut args, flag)?),
            "--height" => height = Some(number(&mut args, flag)?),
            "--start-index" => start_index = number(&mut args, flag)?,
            "--count" => count = number(&mut args, flag)?,
            "--recipe" => {
                recipe_path = Some(PathBuf::from(
                    args.next().context("--recipe requires a JSON file path")?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().context("--output requires a path")?,
                ))
            }
            "--scene-only" => scene_only = true,
            _ => bail!("unknown generation argument: {flag}; use --generate --help"),
        }
    }
    ensure!(count > 0, "--count must be greater than zero");
    let last_index = start_index
        .checked_add(count - 1)
        .context("sample index range exceeds u32::MAX")?;
    ensure!(!scene_only || count == 1, "--scene-only requires --count 1");
    ensure!(
        scene_only || output.is_some(),
        "capture generation requires --output DIRECTORY"
    );
    if let Some(path) = &output {
        ensure!(
            path.file_name().is_some(),
            "output must name a new file or directory"
        );
        require_absent(path)?;
    }
    let recipe = if let Some(path) = recipe_path {
        ensure!(
            seed.is_none() && width.is_none() && height.is_none(),
            "--recipe cannot be combined with --seed, --width or --height"
        );
        serde_json::from_reader::<_, GeneratorRecipe>(
            fs::File::open(&path).with_context(|| format!("opening recipe {}", path.display()))?,
        )
        .with_context(|| format!("parsing recipe {}", path.display()))?
    } else {
        let mut recipe = GeneratorRecipe::default();
        if let Some(seed) = seed {
            recipe.seed = seed;
        }
        if let Some(width) = width {
            recipe.width = width;
        }
        if let Some(height) = height {
            recipe.height = height;
        }
        recipe
    };
    // Validate every realized sample before opening a GPU or writing capture payloads.
    // Regeneration during capture keeps memory bounded for large batches.
    for sample_index in start_index..=last_index {
        recipe.generate(sample_index)?.validate()?;
    }
    if scene_only {
        let scene = recipe.generate(start_index)?;
        let mut bytes = serde_json::to_vec_pretty(&scene)?;
        bytes.push(b'\n');
        if let Some(path) = output {
            let parent = prepare_destination(&path)?;
            let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
            temporary.write_all(&bytes)?;
            temporary.flush()?;
            temporary
                .persist_noclobber(&path)
                .with_context(|| format!("publishing scene {}", path.display()))?;
        } else {
            io::stdout().lock().write_all(&bytes)?;
        }
        return Ok(());
    }
    let output = output.context("capture generation requires --output DIRECTORY")?;
    let parent = prepare_destination(&output)?;
    let staging = tempfile::Builder::new()
        .prefix(".sim-dataset-")
        .tempdir_in(parent)?;
    let mut inspector = Inspector::new().await?;
    let mut samples = Vec::new();
    for sample_index in start_index..=last_index {
        let scene = recipe.generate(sample_index)?;
        let directory = format!("sample-{sample_index:010}");
        inspector
            .capture(&scene)?
            .save(&scene, staging.path().join(&directory))?;
        samples.push(Sample {
            sample_index,
            scene: format!("{directory}/scene.json"),
            metadata: format!("{directory}/metadata.json"),
            directory,
        });
    }
    let manifest = serde_json::json!({
        "version": 1,
        "recipe": recipe,
        "ontology": {
            "version": 1,
            "classes": [
                {"id": 0, "name": "background"},
                {"id": SemanticClass::Ground.id(), "name": "ground"},
                {"id": SemanticClass::Gate.id(), "name": "gate"},
                {"id": SemanticClass::Obstacle.id(), "name": "obstacle"}
            ]
        },
        "start_index": start_index,
        "count": count,
        "samples": samples
    });
    fs::write(
        staging.path().join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    publish(staging.path(), &output)
        .with_context(|| format!("publishing dataset {}", output.display()))?;
    println!("Saved {count} synchronized samples to {}", output.display());
    Ok(())
}

fn number(args: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<u32> {
    let value = args
        .next()
        .with_context(|| format!("{flag} requires an unsigned 32-bit integer"))?;
    let value = value
        .to_str()
        .with_context(|| format!("{flag} must be UTF-8 digits"))?;
    ensure!(
        !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
        "{flag} must be an unsigned 32-bit integer"
    );
    value
        .parse()
        .with_context(|| format!("{flag} exceeds u32::MAX"))
}

fn require_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("checking destination {}", path.display()))
        }
        Ok(_) => bail!(
            "destination {} already exists; refusing to overwrite",
            path.display()
        ),
    }
}

fn prepare_destination(path: &Path) -> Result<&Path> {
    ensure!(
        path.file_name().is_some(),
        "output must name a new file or directory"
    );
    require_absent(path)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("creating output parent {}", parent.display()))?;
    Ok(parent)
}

fn publish(source: &Path, destination: &Path) -> Result<()> {
    require_absent(destination)?;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        source,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    // Standard rename never replaces a populated dataset directory. On other Unix
    // systems, a concurrently created empty directory could replace the precheck's
    // absent path; no existing dataset contents can be overwritten in that race.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fs::rename(source, destination)?;
    Ok(())
}
