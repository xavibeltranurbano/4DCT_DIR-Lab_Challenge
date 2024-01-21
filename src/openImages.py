# # -----------------------------------------------------------------------------
# # File to read raw images using simple itk
# # based on https://simpleitk.readthedocs.io/en/master/link_RawImageReading_docs.html
# # Author: zivy and blowekamp
# # Date Created: 19-03-2020
# # -----------------------------------------------------------------------------

import argparse
import os
import tempfile
import SimpleITK as sitk


def read_raw(
    binary_file_name,
    image_size,
    sitk_pixel_type,
    image_spacing=None,
    image_origin=None,
    big_endian=False,
):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]
    dim = len(image_size)
    header = [
        "ObjectType = Image\n".encode(),
        (f"NDims = {dim}\n").encode(),
        (
            "DimSize = " + " ".join([str(v) for v in image_size]) + "\n"
        ).encode(),
        (
            "ElementSpacing = "
            + (
                " ".join([str(v) for v in image_spacing])
                if image_spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in image_origin])
                if image_origin
                else " ".join(["0"] * dim) + "\n"
            )
        ).encode(),
        ("TransformMatrix = " + direction_cosine[dim - 2] + "\n").encode(),
        ("ElementType = " + pixel_dict[sitk_pixel_type] + "\n").encode(),
        "BinaryData = True\n".encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        # ElementDataFile must be the last entry in the header
        (
            "ElementDataFile = " + os.path.abspath(binary_file_name) + "\n"
        ).encode(),
    ]
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)

    print(header)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img

def main():
    # List of parameters for each image case
    image_cases = [
        {"raw_file_name": "data/copd1/copd1_eBHCT.img", "out_file_name": "data/copd1/copd1_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 121], "spacing": ["0.625", "0.625", "2.5"]},
        {"raw_file_name": "data/copd1/copd1_iBHCT.img", "out_file_name": "data/copd1/copd1_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 121], "spacing": ["0.625", "0.625", "2.5"]},
        
        {"raw_file_name": "data/copd2/copd2_eBHCT.img", "out_file_name": "data/copd2/copd2_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 102], "spacing": ["0.645", "0.645", "2.5"]},
        {"raw_file_name": "data/copd2/copd2_iBHCT.img", "out_file_name": "data/copd2/copd2_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 102], "spacing": ["0.645", "0.645", "2.5"]},
        
        {"raw_file_name": "data/copd3/copd3_eBHCT.img", "out_file_name": "data/copd3/copd3_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 126], "spacing": ["0.652", "0.652", "2.5"]},
        {"raw_file_name": "data/copd3/copd3_iBHCT.img", "out_file_name": "data/copd3/copd3_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 126], "spacing": ["0.652", "0.652", "2.5"]},
        
        {"raw_file_name": "data/copd4/copd4_eBHCT.img", "out_file_name": "data/copd4/copd4_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 126], "spacing": ["0.590", "0.590", "2.5"]},
        {"raw_file_name": "data/copd4/copd4_iBHCT.img", "out_file_name": "data/copd4/copd4_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 126], "spacing": ["0.590", "0.590", "2.5"]},

        # TEST SET
        {"raw_file_name": "data/copd5/copd5_eBHCT.img", "out_file_name": "data/copd5/copd5_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 131], "spacing": ["0.647", "0.647", "2.5"]},
        {"raw_file_name": "data/copd5/copd5_iBHCT.img", "out_file_name": "data/copd5/copd5_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 131], "spacing": ["0.647", "0.647", "2.5"]},

        {"raw_file_name": "data/copd6/copd6_eBHCT.img", "out_file_name": "data/copd6/copd6_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 119], "spacing": ["0.633", "0.633", "2.5"]},
        {"raw_file_name": "data/copd6/copd6_iBHCT.img", "out_file_name": "data/copd6/copd6_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [512, 512, 119], "spacing": ["0.633", "0.633", "2.5"]},

    
        {"raw_file_name": "data/copd0/copd0_eBHCT.img", "out_file_name": "data/copd0/copd0_eBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [256, 256, 94], "spacing": ["0.97", "0.97", "2.5"]},
        {"raw_file_name": "data/copd0/copd0_iBHCT.img", "out_file_name": "data/copd0/copd0_iBHCT.nii", "big_endian": False, "sitk_pixel_type": sitk.sitkInt16, "sz": [256, 256, 94], "spacing": ["0.97", "0.97", "2.5"]},

    ]

    for case in image_cases:
        image = read_raw(
            binary_file_name=case["raw_file_name"],
            image_size=case["sz"],
            sitk_pixel_type=case["sitk_pixel_type"],
            big_endian=case["big_endian"],
            image_spacing=case["spacing"]
        )

        sitk.WriteImage(image, case["out_file_name"])

if __name__ == "__main__":
    main()
