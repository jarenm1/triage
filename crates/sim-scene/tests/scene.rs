use sim_scene::{GeneratorRecipe, SceneConfig, SemanticClass};

#[test]
fn seeking_and_reordered_generation_replay_identically() -> anyhow::Result<()> {
    let recipe = GeneratorRecipe {
        seed: u32::MAX,
        ..Default::default()
    };
    let indices = [0, 1, 97, u32::MAX];
    let scenes = indices.map(|sample| recipe.generate(sample).unwrap());
    for index in (0..indices.len()).rev() {
        let scene = recipe.generate(indices[index])?;
        assert_eq!(scene, scenes[index]);
        let decoded_recipe: GeneratorRecipe =
            serde_json::from_slice(&serde_json::to_vec(&recipe)?)?;
        assert_eq!(decoded_recipe.generate(indices[index])?, scene);
    }
    assert_ne!(scenes[0].objects, scenes[1].objects);
    Ok(())
}

#[test]
fn realized_composite_scene_replays_without_generator_provenance() -> anyhow::Result<()> {
    let mut scene = GeneratorRecipe::default().generate(12)?;
    let original = scene.objects.clone();
    scene.generation = None;
    let temp = tempfile::tempdir()?;
    scene.save(temp.path().join("scene.json"))?;
    let replay = SceneConfig::load(temp.path().join("scene.json"))?;
    assert_eq!(replay, scene);
    assert_eq!(replay.objects, original);
    let gates: Vec<_> = replay
        .objects
        .iter()
        .filter(|object| object.class == SemanticClass::Gate)
        .collect();
    assert_eq!(gates.len(), 2);
    for gate in gates {
        assert_eq!(gate.parts.len(), 3);
        assert!(gate.id > i32::MAX as u32);
    }
    assert!(
        replay
            .objects
            .iter()
            .any(|object| object.class == SemanticClass::Ground && object.id != 0)
    );
    Ok(())
}

#[test]
fn unsupported_scene_and_recipe_versions_are_rejected() -> anyhow::Result<()> {
    let mut scene = GeneratorRecipe::default().generate(0)?;
    scene.version = 1;
    assert!(scene.validate().is_err());
    let mut recipe = GeneratorRecipe::default();
    recipe.version += 1;
    assert!(recipe.generate(0).is_err());
    scene.version = sim_scene::SCENE_VERSION;
    scene.generation.as_mut().unwrap().recipe = recipe;
    assert!(scene.validate().is_err());
    Ok(())
}

#[test]
fn composite_identity_and_geometry_validation_boundaries() -> anyhow::Result<()> {
    let mut scene = SceneConfig::default();
    scene.objects[0].id = u32::MAX;
    let second_part = scene.objects[1].parts[0].clone();
    scene.objects[0].parts.push(second_part);
    scene.validate()?;
    let valid = scene.clone();
    scene.objects[1].id = u32::MAX;
    assert!(scene.validate().is_err());
    scene = valid.clone();
    scene.objects[0].id = 0;
    assert!(scene.validate().is_err());
    scene = valid.clone();
    scene.objects[0].parts.clear();
    assert!(scene.validate().is_err());
    scene = valid.clone();
    scene.objects[0].parts[1].scale[0] = 0.0;
    assert!(scene.validate().is_err());
    scene = valid.clone();
    scene.objects[0].parts[1].position[2] = f32::INFINITY;
    assert!(scene.validate().is_err());
    scene = valid;
    scene.sensor.up =
        (glam::Vec3::from(scene.sensor.target) - glam::Vec3::from(scene.sensor.eye)).to_array();
    assert!(scene.validate().is_err());
    Ok(())
}

#[test]
fn recipe_bounds_fail_before_generation_and_extremes_remain_valid() -> anyhow::Result<()> {
    let recipe = GeneratorRecipe {
        obstacle_count: [24, 24],
        obstacle_size_mm: [1800, 1800],
        obstacle_spread_mm: 6000,
        gate_width_mm: [4000, 4000],
        gate_height_mm: [4200, 4200],
        camera_jitter_mm: 2000,
        ..Default::default()
    };
    for sample in [0, 1, u32::MAX] {
        recipe.generate(sample)?.validate()?;
    }
    let invalid = GeneratorRecipe {
        obstacle_count: [25, 25],
        ..recipe.clone()
    };
    assert!(invalid.generate(0).is_err());
    let invalid = GeneratorRecipe {
        obstacle_size_mm: [1400, 400],
        ..recipe.clone()
    };
    assert!(invalid.generate(0).is_err());
    let invalid = GeneratorRecipe {
        camera_jitter_mm: u32::MAX,
        ..recipe
    };
    assert!(invalid.generate(0).is_err());
    Ok(())
}
