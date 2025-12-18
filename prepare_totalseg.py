import os
import argparse
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import traceback

#############################
# HELPERS
#############################
def reorient_to(img, axcodes_to=("L", "A", "S"), verb=False):
    aff = img.affine
    ornt_fr = nio.io_orientation(aff)
    axcodes_fr = nio.ornt2axcodes(ornt_fr)
    if axcodes_to == axcodes_fr:
        return img

    ornt_to = nio.axcodes2ornt(axcodes_to)
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)

    if verb:
        print(f"[*] Image reoriented from {axcodes_fr} to {axcodes_to}")
    return newimg


def process_subject(subj, base_dir, savepath_root, organ, new_orientation=("L", "A", "S")):
    ct_path = os.path.join(base_dir, subj, "ct.nii.gz")
    seg_path = os.path.join(base_dir, subj, "segmentations", f"{organ}.nii.gz")

    if not os.path.exists(ct_path) or not os.path.exists(seg_path):
        print(ct_path, seg_path)
        return f"Skipped {subj}: Missing files"

    seg_nib = nib.load(seg_path)
    seg_arr = seg_nib.get_fdata(dtype=np.float32)
    if np.max(seg_arr) != 1:
        return f"Skipped {subj}: Max segmentation value != 1"

    img_nib = nib.load(ct_path)
    arr = img_nib.get_fdata(dtype=np.float32)
    zooms = img_nib.header.get_zooms()[:3]

    # Clean affine: keep spacing, discard original orientation/translation
    new_affine = np.eye(4, dtype=np.float32)
    new_affine[0, 0] = zooms[0]
    new_affine[1, 1] = zooms[1]
    new_affine[2, 2] = zooms[2]
    new_affine[:3, 3] = 0

    new_img = nib.Nifti1Image(arr, affine=new_affine)
    new_seg = nib.Nifti1Image(seg_arr, affine=new_affine)

    img_nib_reoriented = reorient_to(new_img, axcodes_to=new_orientation)
    seg_nib_reoriented = reorient_to(new_seg, axcodes_to=new_orientation)

    savepath_img = os.path.join(savepath_root, f"{subj}_img.nii.gz")
    savepath_seg = os.path.join(savepath_root, f"{subj}_msk.nii.gz")

    nib.save(img_nib_reoriented, savepath_img)
    nib.save(seg_nib_reoriented, savepath_seg)

    return f"Processed {subj}: Success"


#############################
# ARGPARSE
#############################
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract an organ from TotalSegmentator and prepare nnU-Net dataset structure."
    )

    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to TotalSegmentator dataset root (contains sXXXX folders)",
    )
    p.add_argument(
        "--nnunet_raw",
        type=str,
        required=True,
        help="Path to nnUNet_raw directory (dataset folder will be created inside)",
    )

    p.add_argument(
        "--organ",
        type=str,
        default="Pancreas",
        help='Organ mask filename (without .nii.gz), e.g. "Pancreas"',
    )
    p.add_argument(
        "--dataset_id",
        type=int,
        default=1,
        help="nnU-Net dataset id (integer). Will become DatasetXXX_*",
    )

    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of subjects used for test set",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )

    p.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Temporary directory for filtered files. Default: <base_dir>/<DatasetName>_filtered",
    )

    p.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete temporary filtered folder after copying to nnU-Net structure",
    )

    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Debug: only process the first N subjects",
    )

    return p.parse_args()


#############################
# MAIN PIPELINE
#############################
def main():
    args = parse_args()

    organ = args.organ
    dataset_name_and_id = f"Dataset{args.dataset_id:03d}_TotalSegmentator{organ.capitalize()}"
    
    organ = args.organ.lower() # Make sure first letter is lowercase

    base_dir = args.base_dir
    nnunet_root = os.path.join(args.nnunet_raw, dataset_name_and_id)

    # Default tmp_dir next to the dataset
    savepath_root = (
        args.tmp_dir
        if args.tmp_dir is not None
        else os.path.join(args.base_dir, f"{dataset_name_and_id}_filtered")
    )

    os.makedirs(savepath_root, exist_ok=True)

    subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("s")])
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    print(f"Step 1/2: Processing {len(subjects)} subjects (single process)...")
    results = []

    for subj in tqdm(subjects, desc="Processing subjects"):
        try:
            msg = process_subject(subj, base_dir, savepath_root, organ)
            results.append(msg)
        except Exception as e:
            print(f"\nError processing {subj}: {e}")
            traceback.print_exc()
            results.append(f"Error processing {subj}: {str(e)}")

    success_count = sum(1 for r in results if "Success" in r)
    skip_count = sum(1 for r in results if "Skipped" in r)
    error_count = sum(1 for r in results if "Error" in r)
    print(f"Finished preprocessing. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    print("\nStep 2/2: Creating nnU-Net folder structure and splitting train/test...")

    output_image_tr_path = os.path.join(nnunet_root, "imagesTr")
    output_label_tr_path = os.path.join(nnunet_root, "labelsTr")
    output_image_ts_path = os.path.join(nnunet_root, "imagesTs")
    output_label_ts_path = os.path.join(nnunet_root, "labelsTs")

    os.makedirs(output_image_tr_path, exist_ok=True)
    os.makedirs(output_label_tr_path, exist_ok=True)
    os.makedirs(output_image_ts_path, exist_ok=True)
    os.makedirs(output_label_ts_path, exist_ok=True)

    all_subjects = sorted({x.split("_")[0] for x in os.listdir(savepath_root) if x.endswith("img.nii.gz")})
    train_subjects, test_subjects = train_test_split(
        all_subjects, test_size=args.test_size, random_state=args.seed
    )

    print(f"Number of training subjects: {len(train_subjects)}")
    print(f"Number of test subjects: {len(test_subjects)}")

    def copy_files(subjects_list, image_dest, label_dest):
        for subject in tqdm(subjects_list, desc=f"Copying to {os.path.basename(image_dest)}"):
            image_src = os.path.join(savepath_root, f"{subject}_img.nii.gz")
            label_src = os.path.join(savepath_root, f"{subject}_msk.nii.gz")

            subject_number = str(int(subject[1:]))  # "s1234" -> "1234"
            image_dst = os.path.join(image_dest, f"{subject_number}_0000.nii.gz")
            label_dst = os.path.join(label_dest, f"{subject_number}.nii.gz")

            if os.path.exists(image_src) and os.path.exists(label_src):
                shutil.copy2(image_src, image_dst)
                shutil.copy2(label_src, label_dst)
            else:
                print(f"Missing files for subject {subject}")

    copy_files(train_subjects, output_image_tr_path, output_label_tr_path)
    copy_files(test_subjects, output_image_ts_path, output_label_ts_path)

    if args.cleanup:
        shutil.rmtree(savepath_root)

    print("\nAll done! Dataset prepared for nnU-Net.")
    print(f"Temporary folder: {savepath_root}")
    print(f"nnU-Net dataset folder: {nnunet_root}")


if __name__ == "__main__":
    main()
