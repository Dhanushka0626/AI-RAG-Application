from typing import Any, Dict

from PIL import Image
from img2vec_pytorch import Img2Vec
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import register_embedding_function

DB_PATH = "./db"
DATA_PATH = "./data"

@register_embedding_function
class ImageEmbeddingFunction(EmbeddingFunction):

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        embeddings = self._get_imgs_embeddings(input)
        return embeddings

    def _get_imgs_embeddings(self, input):
        embeddings = []

        for img in input:
            if img is None:
                raise ValueError("Image could not be loaded. Check image path or file extension.")

            if isinstance(img, str):
                image = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                image = Image.fromarray(img).convert("RGB")

            embeddings.append(self.model.get_vec(image).tolist())

        return embeddings
    
    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddingFunction(config['model'])