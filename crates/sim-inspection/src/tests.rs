use super::*;
use sim_scene::SemanticClass;

fn fixture() -> (SceneConfig, Capture) {
    let mut config = SceneConfig::default();
    config.sensor.width = 3;
    config.sensor.height = 1;
    config.objects[0].id = u32::MAX;
    config.objects[0].class = SemanticClass::Gate;
    let second = config.objects[1].parts[0].clone();
    config.objects[0].parts.push(second);
    let capture = Capture {
        preview: RgbaImage::new(3, 1),
        color: vec![0, 128, 255, 255, 64, 0, 0, 255, 0, 0, 0, 255],
        depth: vec![2.0, 2.5, config.sensor.far],
        object_ids: vec![u32::MAX, u32::MAX, 0],
        config: config.clone(),
    };
    (config, capture)
}

#[test]
fn composite_instance_mask_exports_classes_and_same_source_srgb() -> Result<()> {
    let (config, capture) = fixture();
    let temp = tempfile::tempdir()?;
    let bundle = temp.path().join("capture");
    capture.save(&config, &bundle)?;
    let ids = fs::read(bundle.join("object_ids.u32le"))?;
    assert_eq!(
        ids,
        [u32::MAX, u32::MAX, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let classes = fs::read(bundle.join("semantic_classes.u32le"))?;
    assert_eq!(
        classes,
        [2_u32, 2, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(fs::read(bundle.join("color.rgba8"))?, capture.color);
    let png = image::open(bundle.join("color.png"))?.into_rgba8();
    assert_eq!(png.get_pixel(0, 0).0, [0, 188, 255, 255]);
    assert_eq!(SceneConfig::load(bundle.join("scene.json"))?, config);
    assert!(capture.save(&config, &bundle).is_err());
    assert_eq!(fs::read(bundle.join("object_ids.u32le"))?, ids);
    Ok(())
}

#[test]
fn unknown_instance_or_mismatched_scene_never_publishes_bundle() -> Result<()> {
    let (mut config, mut capture) = fixture();
    let temp = tempfile::tempdir()?;
    let bundle = temp.path().join("capture");
    capture.object_ids[0] = 42;
    assert!(capture.save(&config, &bundle).is_err());
    assert!(!bundle.exists());
    capture.object_ids[0] = u32::MAX;
    config.objects[0].class = SemanticClass::Obstacle;
    assert!(capture.save(&config, &bundle).is_err());
    assert!(!bundle.exists());
    Ok(())
}
