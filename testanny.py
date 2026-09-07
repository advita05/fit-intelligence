import numpy as np, torch, anny
from clad_body.load import AnnyBody
from clad_body.measure import measure

# Create the base Anny model once, with all phenotype controls enabled
anny_model = anny.create_fullbody_model(
    all_phenotypes=True, triangulate_faces=True, local_changes=False
).to(dtype=torch.float32)

# Height range Anny's model was calibrated on, per gender (used to normalize input)
HEIGHT_RANGE_BY_GENDER = {
    0.0: (133.4, 242.28),
    1.0: (121.07, 230.08),
}

def normalize_height(height_cm, gender):
    # Convert real height (cm) into Anny's expected 0-1 scale using min-max normalization
    g = 0.0 if gender < 0.5 else 1.0
    h_min, h_max = HEIGHT_RANGE_BY_GENDER[g]
    return max(0.0, min(1.0, (height_cm - h_min) / (h_max - h_min)))


def _build_body(height_norm, weight_norm, gender, age_norm):
    # Package phenotype inputs as PyTorch tensors (required format for the model)
    pk = {
        "height": torch.tensor([height_norm]),
        "weight": torch.tensor([weight_norm]),
        "gender": torch.tensor([float(gender)]),
        "age": torch.tensor([age_norm]),
    }

    # Neutral standing pose - no rotation applied to any joint
    identity_pose = torch.eye(4)[None, None].repeat(1, anny_model.bone_count, 1, 1)

    # Generate the 3D mesh (no_grad = skip gradient tracking, we're not training)
    with torch.no_grad():
        output = anny_model(
            pose_parameters=identity_pose,
            phenotype_kwargs=pk,
            pose_parameterization="local-ref",
            return_bone_ends=True,
        )

    # Store bone/phenotype data on the model - clad_body reads this internally
    anny_model._last_bone_heads = output.get("bone_heads")
    anny_model._last_bone_tails = output.get("bone_tails")
    anny_model._last_phenotype_kwargs = pk

    # Extract mesh vertices and faces as plain NumPy arrays
    v = output["vertices"][0].cpu().numpy().astype(np.float32)
    f = anny_model.faces.cpu().numpy().astype(np.int32)

    # Fix axis orientation: ensure "height" consistently maps to the z-axis
    if int(np.argmax(v.max(0) - v.min(0))) == 1:
        v = v[:, [0, 2, 1]].copy()
        v[:, 2] = -v[:, 2]

    # Place feet on the ground (z=0) and center the body horizontally
    v[:, 2] -= v[:, 2].min()
    c = (v[:, :2].max(0) + v[:, :2].min(0)) / 2
    v[:, 0] -= c[0]
    v[:, 1] -= c[1]

    # Package into the format clad_body's measure() function expects
    return AnnyBody(
        vertices=v, faces=f, source="manual", phenotype_params=pk, _model=anny_model
    )


def _find_weight_norm(height_norm, target_kg, gender, age_norm, iters=25):
    # Binary search: Anny's "weight" input is normalized (0-1), not real kg,
    # so we search for the normalized value that produces our target real weight
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        body = _build_body(height_norm, mid, gender, age_norm)
        mass = measure(body, only=["mass_kg"])["mass_kg"]
        if mass < target_kg:
            lo = mid  # too light, search upper half
        else:
            hi = mid  # too heavy, search lower half
    return mid


def get_anny_measurements(height_cm, weight_kg, gender, age):
    # Full pipeline: real inputs -> normalized inputs -> 3D body -> ISO measurements
    height_norm = normalize_height(height_cm, gender)
    age_norm = 1.0  # placeholder: age not yet calibrated to real years
    weight_norm = _find_weight_norm(height_norm, weight_kg, gender, age_norm)
    body = _build_body(height_norm, weight_norm, gender, age_norm)
    return measure(body, only=["height_cm", "bust_cm", "waist_cm", "hip_cm", "mass_kg"])


if __name__ == "__main__":
    height_cm = 152
    weight_kg = 48
    gender = 1.0
    age = 25

    height_norm = normalize_height(height_cm, gender)
    age_norm = 1.0
    weight_norm = _find_weight_norm(height_norm, weight_kg, gender, age_norm)
    body = _build_body(height_norm, weight_norm, gender, age_norm)

    # Requesting specific keys only 
    desired_keys = [
        "height_cm", "bust_cm", "waist_cm", "hip_cm", "underbust_cm",
        "neck_cm", "stomach_cm", "mass_kg"
    ]

    results = measure(body, only=desired_keys)

    print("---- ANNY MEASUREMENTS (ISO 8559-1) ----")
    for key in desired_keys:
        if key in results:
            print(f"{key:<15}: {results[key]:.2f}")
        else:
            print(f"{key:<15}: NOT AVAILABLE")