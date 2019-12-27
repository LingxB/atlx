from src.models.atlstm import ATLSTM
from src.models.atlstm_loss_sum import ATLSTM_loss_sum
from src.models.atlx import ATLX

VALID_MODELS = dict(
    atlstm=ATLSTM,
    atlstm_loss_sum=ATLSTM_loss_sum,
    atlx=ATLX
)