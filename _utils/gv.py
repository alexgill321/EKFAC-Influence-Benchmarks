import glob
import os
import glob
import os
import re
import warnings
from typing import Any, List, Optional, Tuple, Union

import captum._utils.common as common
import torch
from captum.attr import LayerActivation
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class GV:
    r"""
    
    Shell for providing functionality for storing and loading gradients for
    pre-defined neural network layers.
    It also provides functionality to check if gradients already exist in the
    manifold and other auxiliary functions.

    This class also defines a torch `Dataset`, representing Gradient Vectors,
    which enables lazy access to gradients and layer stored in the manifold.
    """

    class GVDataset(Dataset):
        r"""
        This dataset enables access to gradient vectors for a given `model` stored
        under a pre-defined path.
        The iterator of this dataset returns a batch of data tensors.
        Additionally, subsets of the model gradients can be loaded based on layer
        or identifier or num_id (representing batch number in source dataset).
        """

        def __init__(
            self,
            path: str,
            model_id: str,
            identifier: Optional[str] = None,
            layer: Optional[str] = None,
            num_id: Optional[str] = None,
        ) -> None:
            
            self.gv_filesearch = GV._construct_file_search(
                path, model_id, identifier, layer, num_id
            )

            files = glob.glob(self.gv_filesearch)

            self.files = GV.sort_files(files)
        
        def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, ...]]:
            assert idx < len(self.files), "Layer index is out of bounds!"
            fl = self.files[idx]
            gv = torch.load(fl)
            return gv
        
        def __len__(self) -> int:
            return len(self.files)

    GV_DIR_NAME: str = "gv"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _assemble_model_dir(path: str, model_id: str) -> str:
        r"""
        Returns a directory path for the given source path `path` and `model_id.`
        This path is suffixed with the '/' delimiter.
        """
        return "/".join([path, GV.GV_DIR_NAME, model_id, ""])

    @staticmethod
    def _construct_file_search(
        source_dir: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> str:
        r"""
        Returns a search string that can be used by glob to search `source_dir/model_id`
        for the desired layer/identifier pair. Leaving `layer` as None will search ids
        over all layers, and leaving `identifier` as none will search layers over all
        ids.  Leaving both as none will return a path to glob for every activation.
        Assumes identifier is always specified when saving activations, so that
        activations live at source_dir/model_id/identifier/layer
        (and never source_dir/model_id/layer)
        """

        gv_filesearch = GV._assemble_model_dir(source_dir, model_id)

        gv_filesearch = os.path.join(
            gv_filesearch, "*" if identifier is None else identifier
        )

        gv_filesearch = os.path.join(
            gv_filesearch, "*" if layer is None else layer
        )

        gv_filesearch = os.path.join(
            gv_filesearch, "*.pt" if num_id is None else "%s.pt" % num_id
        )

        return gv_filesearch
    
    @staticmethod
    def sort_files(files: List[str]) -> List[str]:
        r"""
        Utility for sorting files based on natural sorting instead of the default
        lexigraphical sort.
        """

        def split_alphanum(s):
            r"""
            Splits string into a list of strings and numbers
                "z23a" -> ["z", 23, "a"]
            """

            return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]
        
        return sorted(files, key=split_alphanum)

    @staticmethod
    def exists(

        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> bool:
        r"""
        Verifies whether the model + layer gradients exist under the path.

        Args:
            path (str): The path where the gradient vectors
                    for the `model_id` are stored.
            model_id (str): The name/version of the model for which layer gradients
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer gradients.
                    Can be used to distinguish between gradients for different
                    training batches.
            layer (str or None): The layer for which the gradient vectors are computed
            num_id (str): An optional string representing the batch number for which
                the gradient vectors are computed.

        Returns:
            exists (bool): Indicating whether the gradient vectors for the `layer`
            and `identifier` (if provided) and num_id (if provided) were stored in
            the manifold. If no `identifier` is provided, will return `True` if any
            gradient vectors for the `layer` exist.
        """
        gv_dir = GV._assemble_model_dir(path, model_id)
        gv_filesearch = GV._construct_file_search(
            path, model_id, identifier, layer, num_id
        )
        return os.path.exists(gv_dir) and len(glob.glob(gv_filesearch)) > 0
    
    @staticmethod
    def load(
        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> GVDataset:
        r"""
        Loads the gradient vectors for the given `model_id` and
        `layer` saved under the `path`.

        Args:
            path (str): The path where the gradient vectors
                    for the `model_id` are stored.
            model_id (str): The name/version of the model for which layer gradients
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer gradients.
                    Can be used to distinguish between gradients for different
                    training batches.
            layer (str or None): The layer for which the gradient vectors are computed
            num_id (str): An optional string representing the batch number for which
                the gradient vectors are computed.

        Returns:
            dataset (GV.GVDataset): The dataset that allows iteration over the 
                gradient vectors for the given layer, identifier (if provided),
                and num_id (if provided). Returning an GV.GVDataset as opposed to a
                dataloader constructed from it offers more flexibility. Raises RuntimeError
                if gradient vectors are not found.
        """

        gv_save_dir = GV._assemble_model_dir(path, model_id)

        if os.path.exists(gv_save_dir):
            gvdataset = GV.GVDataset(path, model_id, identifier, layer, num_id)
            return gvdataset
        else:
            raise RuntimeError(
                f"Gradient vectors for model {model_id} was not found at path {path}"
            )
        
    @staticmethod
    def _manage_loading_layers(
        path: str,
        model_id: str,
        layers: Union[str, List[str]],
        load_from_disk: bool = True,
        identifier: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> List[str]:
        r"""
        Returns unsaved layers, and deletes saved layers if load_from_disk is False.

        Args:
            path (str): The path where the gradient vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer gradients
                    are being computed and stored.
            layers (str or List[str]): The layers for which the gradient vectors
                are computed.
            load_from_disk (bool): Whether to load the gradient vectors from disk.
            identifier (str or None): An optional identifier for the layer gradients.
                    Can be used to distinguish between gradients for different
                    training batches.
            num_id (str): An optional string representing the batch number for which
                    the gradient vectors are computed.

        Returns:
            List of layer names for which gradient vectors should be generated
        """
        layers = [layers] if isinstance(layers, str) else layers
        unsaved_layers = []

        if load_from_disk:
            for layer in layers:
                if not GV.exists(path, model_id, identifier, layer, num_id):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = layers
            warnings.warn(
                "Overwriting gradients: load_from_disk is set to False. Removing all "
                f"gradients matching specified parameters {{path: {path}, "
                f"model_id: {model_id}, layers: {layers}, identifier: {identifier}}} "
                "before generating new gradients."
            )
            for layer in layers:
                files = glob.glob(
                    GV._construct_file_search(path, model_id, identifier, layer)
                )
                for filename in files:
                    os.remove(filename)

        return unsaved_layers
    
    @staticmethod
    def _compute_and_save_gradients(
        path: str,
        model: Module,
        model_id: str,
        layers: Union[str, List[str]],
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        identifier: str,
        num_id: Optional[str] = None,
        additional_forward_args: Any = None,
        load_from_disk: bool = True,
    ) -> None:
        r"""
        Computes layer gradients for the given inputs and specified 'layers'

        Args:
            path (str): The path where the gradient vectors
                    for the `layer` are stored.
            model (torch.nn.Module): An instance of pytorch model. This model should
                define all of its layers as attributes of the model.
            model_id (str): The name/version of the model for which layer gradients
                    are being computed and stored.
            layers (str or List[str]): The layers for which the gradient vectors
                are computed.
            inputs (torch.Tensor or Tuple[torch.Tensor, ...]): The inputs to
                the model.
        """

        # TODO: Write this method for influence utils