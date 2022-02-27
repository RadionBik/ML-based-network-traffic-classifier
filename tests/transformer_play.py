from torch.utils.data.dataloader import DataLoader

import settings
from gpt_model.generator.dataset import PretrainDatasetWithClasses, PretrainCollator
from transformers import GPT2Model
from gpt_model.tokenizer import PacketTokenizer


def main():
    raw_dataset_folder = settings.TEST_STATIC_DIR / 'raw_csv'
    checkpoint = '/media/raid_store/pretrained_traffic/gpt2_model_4epochs_classes_external'
    tokenizer = PacketTokenizer.from_pretrained(checkpoint, flow_size=20)
    model = GPT2Model.from_pretrained(checkpoint)
    ds = PretrainDatasetWithClasses(tokenizer, folder_path=raw_dataset_folder)
    loader = DataLoader(ds, batch_size=4, collate_fn=PretrainCollator(tokenizer), drop_last=True)
    for flow in loader:
        assert flow['input_ids'].shape == (4, 22)
        # 9905 is the last non-flow-label token ID
        assert (flow['input_ids'][:, 0] > 9905).all().tolist()
        flow.pop('labels')
        output = model(**flow)


if __name__ == '__main__':
    main()
