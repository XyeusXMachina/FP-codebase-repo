from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


INPUT_DIR = Path("AFLFP")
OUTPUT_DIR = Path("dataset/normal")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def generate_mesh(image, face_mesh, drawing_utils, face_mesh_module):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    height, width = image.shape[:2]

    mesh_image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for face_landmarks in results.multi_face_landmarks:
        drawing_utils.draw_landmarks(
            image=mesh_image,
            landmark_list=face_landmarks,
            connections=face_mesh_module.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(255, 255, 255),
                thickness=1,
                circle_radius=1,
            ),
        )

    return mesh_image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_dir() and path.name.isdigit()
    )

    image_files = []

    for subject_dir in subject_dirs:
        for image_path in subject_dir.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not image_path.stem.endswith("_lmk")
            ):
                image_files.append(image_path)

    image_files.sort()

    total = len(image_files)
    processed = 0
    skipped = 0

    face_mesh_module = mp.solutions.face_mesh
    drawing_utils = mp.solutions.drawing_utils

    print(f"Subjects found: {len(subject_dirs)}")
    print(f"Clean JPG images found: {total}")
    print()

    with create_face_mesh() as face_mesh:

        for index, image_path in enumerate(image_files, start=1):

            subject_id = image_path.parts[
                image_path.parts.index(INPUT_DIR.name) + 1
            ]

            output_subject_dir = (
                OUTPUT_DIR / f"aflfp_{int(subject_id):03d}"
            )

            output_subject_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_filename = (
                f"{image_path.stem}_mesh.png"
            )

            output_path = output_subject_dir / output_filename

            if output_path.exists():
                processed += 1

                print(
                    f"\rProcessed: {index}/{total}",
                    end="",
                    flush=True,
                )

                continue

            image = cv2.imread(str(image_path))

            if image is None:
                skipped += 1
                continue

            mesh_image = generate_mesh(
                image=image,
                face_mesh=face_mesh,
                drawing_utils=drawing_utils,
                face_mesh_module=face_mesh_module,
            )

            if mesh_image is None:
                skipped += 1
                continue

            cv2.imwrite(
                str(output_path),
                mesh_image,
            )

            processed += 1

            print(
                f"\rProcessed: {index}/{total}",
                end="",
                flush=True,
            )

    print()
    print()
    print("Processing completed.")
    print(f"Subjects: {len(subject_dirs)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()