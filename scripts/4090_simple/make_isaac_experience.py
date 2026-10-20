"""Create a local Isaac4.5 experience for SIMPLE's CuRobo/SONIC path, without Lula."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sysconfig
try:
    import tomllib
except ImportError:
    import tomli as tomllib


# Fixed Isaac4.5 base experience roots which pull in Lula. Do not remove core,
# camera, physics, robot manipulators or the RTX/Replicator dependencies.
EXCLUDED = {
    "isaacsim.cortex.behaviors", "isaacsim.cortex.framework",
    "isaacsim.robot_motion.lula", "isaacsim.robot_motion.motion_generation",
    "omni.isaac.cortex.sample_behaviors", "omni.isaac.cortex", "omni.isaac.franka",
    "omni.isaac.lula_test_widget", "omni.isaac.lula", "omni.isaac.motion_generation",
    "omni.isaac.universal_robots",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaac-root", type=Path,
                   default=Path(sysconfig.get_paths()["purelib"]) / "isaacsim")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.isaac_root.expanduser().resolve()
    source = root / "apps/isaacsim.exp.base.kit"
    text = source.read_text()
    config = tomllib.loads(text)
    assert config["package"]["version"] == "4.5.0", "only the fixed Isaac4.5 experience is supported"
    assert EXCLUDED <= config["dependencies"].keys(), "unexpected Isaac4.5 dependency roots"
    lines, dependencies = [], False
    for line in text.splitlines():
        if line.startswith("["):
            dependencies = line.strip() == "[dependencies]"
        match = re.match(r'^"([^"]+)"\s*=', line) if dependencies else None
        if match and match.group(1) in EXCLUDED:
            continue
        lines.append(line)
    generated = "\n".join(lines) + "\n"
    # The copied .kit is outside site-packages; all original app-relative asset,
    # extension, VERSION and registry paths must still resolve inside Isaac4.5.
    escaped_root = str(root).replace("\\", "\\\\").replace('"', '\\"')
    generated = generated.replace("${app}/..", escaped_root)
    generated = generated.replace('name = "Isaac-Sim"', 'name = "CVPR-SIMPLE"')
    generated = generated.replace('title = "Isaac Sim Base"', 'title = "CVPR SIMPLE (CuRobo)"')
    result = tomllib.loads(generated)
    assert not EXCLUDED.intersection(result["dependencies"])
    assert {"isaacsim.core.api", "isaacsim.sensors.camera", "omni.physx.bundle",
            "omni.hydra.rtx", "omni.replicator.core"} <= result["dependencies"].keys()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated)
    metadata = {"isaac_root": str(root), "source": str(source),
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "output": str(output), "excluded_dependency_roots": sorted(EXCLUDED),
                "scope": "SIMPLE CuRobo/SONIC; does not provide Lula/Cortex/legacy arm examples"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
