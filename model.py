import torch
import torch.nn as nn
from transformers import ResNetModel
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, BertLayer


class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        # text subnets
        #self.aligned = args.need_data_aligned
        #self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        self.text_model = BertPreTrainedModel.from_pretrained('bert-base-uncased')
        self.image_model = ResNetModel.from_pretrained("microsoft/resnet-152")

        # the post_fusion layers
        self.fusion_dropout = nn.Dropout(p=args.fusion_dropout)
        self.fusion_layer_1 = nn.Linear(768+2048, args.fusion_dim)
        self.fusion_layer_2 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.fusion_layer_3 = nn.Linear(args.fusion_dim, 3)

        # the classify layer for text
        self.text_dropout = nn.Dropout(p=args.text_dropout)
        self.text_layer_1 = nn.Linear(768, args.text_dim)
        self.text_layer_2 = nn.Linear(args.text_dim, args.text_dim)
        self.text_layer_3 = nn.Linear(args.text_dim, 3)


        # the classify layer for image
        #self.align_layer = nn.Linear(in_features=2048, out_features=768)
        self.image_dropout = nn.Dropout(p=args.image_dropout)
        self.image_layer_1 = nn.Linear(2048, args.image_dim)
        self.image_layer_2 = nn.Linear(args.image_dim, args.image_dim)
        self.image_layer_3 = nn.Linear(args.image_dim, 3)

    def forward(self, text, image):

        text_encode = self.text_model(text)
        text_hidden_state = text_encode.last_hidden_state
        text_hidden_state = text_hidden_state[:, 0, :]
        image_hidden_state = self.image_model(image).pooler_output
        #image_max_pool, _ = image_encode.max(1)
        #image_hidden_state = self.align_layer(image_max_pool).unsqueeze(1)

        """拼接文本和图像，拼接得到共同特征"""
        image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

        fusion_h = self.post_fusion_dropout(image_text_hidden_state)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)

        # text
        text_h = self.text_dropout(text_hidden_state)
        text_h = F.relu(self.text_layer_1(text_h), inplace=False)

        # vision
        image_h = self.image_dropout(image_hidden_state)
        image_h = F.relu(self.image_layer_1(image_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.text_layer_2(text_h), inplace=False)
        output_text = self.text_layer_3(x_t)

        # classifier-vision
        x_v = F.relu(self.image_layer_2(image_h), inplace=False)
        output_image = self.image_layer_3(x_v)

        out = torch.mean(torch.stack([output_image, output_text, output_fusion], dim=2), dim=2)

        return out