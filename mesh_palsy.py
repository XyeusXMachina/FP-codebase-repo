from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


INPUT_DIR = Path("Palsy Set/Subjects")
OUTPUT_DIR = Path("Palsy Set/mesh/Subjects")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def create_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def generate_mesh_image(image, face_mesh, drawing_utils, face_mesh_module):
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

    image_files = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    face_mesh_module = mp.solutions.face_mesh
    drawing_utils = mp.solutions.drawing_utils

    processed = 0
    skipped = 0

    with create_face_mesh() as face_mesh:
        for index, image_path in enumerate(image_files, start=1):
            relative_path = image_path.relative_to(INPUT_DIR)
            output_folder = OUTPUT_DIR / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)

            output_path = output_folder / f"{image_path.stem}_mesh.png"

            image = cv2.imread(str(image_path))

            if image is None:
                skipped += 1
                continue

            mesh_image = generate_mesh_image(
                image=image,
                face_mesh=face_mesh,
                drawing_utils=drawing_utils,
                face_mesh_module=face_mesh_module,
            )

            if mesh_image is None:
                skipped += 1
                continue

            cv2.imwrite(str(output_path), mesh_image)
            processed += 1

            print(
                f"\rProcessed: {index}/{len(image_files)}",
                end="",
                flush=True,
            )

    print()
    print(f"Completed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()