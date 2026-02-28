"""
DICOM utilities for reading and processing CT scans
"""
import pydicom
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def read_dicom_file(filepath: Path) -> pydicom.Dataset:
    """
    Read a single DICOM file
    
    Args:
        filepath: Path to DICOM file
        
    Returns:
        pydicom.Dataset object
    """
    try:
        ds = pydicom.dcmread(str(filepath), force=True)
        return ds
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def get_hu_values(ds: pydicom.Dataset) -> np.ndarray:
    """
    Extract Hounsfield Unit (HU) values from DICOM
    
    Args:
        ds: pydicom.Dataset object
        
    Returns:
        numpy array of HU values
    """
    # Get pixel array
    pixels = ds.pixel_array.astype(np.float32)
    
    # Apply rescale slope and intercept to get HU values
    intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
    slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
    
    hu_values = pixels * slope + intercept
    
    return hu_values


def get_slice_metadata(ds: pydicom.Dataset) -> Dict:
    """
    Extract relevant metadata from DICOM
    
    Args:
        ds: pydicom.Dataset object
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    # Essential fields
    metadata['SOPInstanceUID'] = str(ds.SOPInstanceUID) if hasattr(ds, 'SOPInstanceUID') else None
    metadata['SeriesInstanceUID'] = str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else None
    metadata['StudyInstanceUID'] = str(ds.StudyInstanceUID) if hasattr(ds, 'StudyInstanceUID') else None
    
    # Patient/Study info
    metadata['PatientID'] = str(ds.PatientID) if hasattr(ds, 'PatientID') else None
    
    # Series description for filtering
    metadata['SeriesDescription'] = str(ds.SeriesDescription) if hasattr(ds, 'SeriesDescription') else ""
    
    # Spatial information
    metadata['InstanceNumber'] = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else None
    metadata['SliceLocation'] = float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None
    
    # Image position (for sorting slices)
    if hasattr(ds, 'ImagePositionPatient'):
        metadata['ImagePositionPatient'] = [float(x) for x in ds.ImagePositionPatient]
        metadata['SlicePosition'] = float(ds.ImagePositionPatient[2])  # Z-coordinate
    else:
        metadata['ImagePositionPatient'] = None
        metadata['SlicePosition'] = None
    
    # Pixel spacing
    if hasattr(ds, 'PixelSpacing'):
        metadata['PixelSpacing'] = [float(x) for x in ds.PixelSpacing]
    else:
        metadata['PixelSpacing'] = [1.0, 1.0]  # Default
    
    # Slice thickness
    metadata['SliceThickness'] = float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else None
    
    # Image dimensions
    metadata['Rows'] = int(ds.Rows) if hasattr(ds, 'Rows') else None
    metadata['Columns'] = int(ds.Columns) if hasattr(ds, 'Columns') else None
    
    return metadata


def load_dicom_series(dicom_files: List[Path]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load a series of DICOM files and sort by slice position
    
    Args:
        dicom_files: List of paths to DICOM files
        
    Returns:
        Tuple of (3D numpy array of HU values, list of metadata dicts)
    """
    slices_data = []
    
    for filepath in dicom_files:
        ds = read_dicom_file(filepath)
        if ds is None:
            continue
            
        hu_values = get_hu_values(ds)
        metadata = get_slice_metadata(ds)
        
        slices_data.append({
            'hu_values': hu_values,
            'metadata': metadata,
            'filepath': filepath
        })
    
    # Sort by slice position
    slices_data.sort(key=lambda x: x['metadata']['SlicePosition'] if x['metadata']['SlicePosition'] is not None else x['metadata']['InstanceNumber'])
    
    # Stack into 3D volume
    volume = np.stack([s['hu_values'] for s in slices_data], axis=-1)  # Shape: (H, W, D)
    metadata_list = [s['metadata'] for s in slices_data]
    
    return volume, metadata_list


def is_non_contrast_series(series_description: str, 
                          include_keywords: List[str],
                          exclude_keywords: List[str]) -> bool:
    """
    Check if a series is non-contrast based on description
    
    Args:
        series_description: DICOM SeriesDescription field
        include_keywords: Keywords that should be present
        exclude_keywords: Keywords that should NOT be present
        
    Returns:
        True if non-contrast, False otherwise
    """
    series_desc_lower = series_description.lower()
    
    # Check for exclude keywords first
    for keyword in exclude_keywords:
        if keyword.lower() in series_desc_lower:
            return False
    
    # Check for include keywords
    for keyword in include_keywords:
        if keyword.lower() in series_desc_lower:
            return True
    
    # Default to True if no exclude keywords found (conservative approach)
    return True


def get_series_from_directory(directory: Path) -> Dict[str, List[Path]]:
    """
    Group DICOM files by SeriesInstanceUID
    
    Args:
        directory: Directory containing DICOM files
        
    Returns:
        Dictionary mapping SeriesInstanceUID to list of file paths
    """
    series_dict = {}
    
    # Find all DICOM files
    dicom_files = list(directory.rglob("*.dcm"))
    
    for filepath in dicom_files:
        ds = read_dicom_file(filepath)
        if ds is None:
            continue
        
        series_uid = str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else "unknown"
        
        if series_uid not in series_dict:
            series_dict[series_uid] = []
        
        series_dict[series_uid].append(filepath)
    
    return series_dict


if __name__ == "__main__":
    # Test the utilities
    print("DICOM utilities module loaded successfully!")
    print("Available functions:")
    print("  - read_dicom_file()")
    print("  - get_hu_values()")
    print("  - get_slice_metadata()")
    print("  - load_dicom_series()")
    print("  - is_non_contrast_series()")
    print("  - get_series_from_directory()")
