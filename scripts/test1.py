from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import os
from table_generation import plot_confusion_matrix_with_text2
from IPython import embed


def main():
    pretrained_model = "Qwen/Qwen2-VL-72B-Instruct"
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model, torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # 冠詞と空白は絶対につけない！
    peg_name = input(
        "Enter the peg name: "
    )  # "3D-printed peg" or "toy peg" or "male connector"
    hole_name = input(
        "Enter the hole name: "
    )  # "3D-printed hole" or "toy hole" or "female connector"

    # folderを準備
    image_folder_peg = (
        "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/peg"
    )
    image_folder_hole = (
        "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/hole"
    )
    image_folder_peg2 = (
        "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/peg2"
    )
    image_folder_hole2 = (
        "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/hole2"
    )
    image_files_peg = sorted(
        [
            f
            for f in os.listdir(image_folder_peg)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    image_files_hole = sorted(
        [
            f
            for f in os.listdir(image_folder_hole)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )

    image_files_peg2 = sorted(
        [
            f
            for f in os.listdir(image_folder_peg2)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    image_files_hole2 = sorted(
        [
            f
            for f in os.listdir(image_folder_hole2)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )

    image_files_peg_label = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(image_folder_peg)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    image_files_hole_label = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(image_folder_hole)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    image_files_peg_label2 = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(image_folder_peg2)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    image_files_hole_label2 = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(image_folder_hole2)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
    )
    
    
    path = Path(image_folder_hole)
    
    top_1_data = []  # 全種類のpegとholeのマッチングデータを記録する変数
    
    # 推論パート
    for i, image_file_peg in enumerate(image_files_peg):
        image_file_peg = os.path.join(image_folder_peg, image_file_peg)
        image_file_peg2 = os.path.join(image_folder_peg2, image_files_peg2[i])
        
        each_peg_matching_result = []  # 1つのpegに対してholeのマッチング結果を記録する変数
        
        for j, image_file_hole in enumerate(image_files_hole):
            image_file_hole = os.path.join(image_folder_hole, image_file_hole)
            image_file_hole2 =  os.path.join(image_folder_hole2, image_files_hole2[j])
            # 画像ファイルをリストに追加
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file_peg,
                        },
                        {
                            "type": "image",
                            "image": image_file_peg2,
                        },
                        {
                            "type": "image",
                            "image": image_file_hole,
                        },
                        {
                            "type": "image",
                            "image": image_file_hole2,
                        },
                        {
                            "type": "text",
                            # "text": "Explain about image 3."
                            "text": f"Image 1 is a cross-sectional image of a {peg_name}. Image 2 is another image of a {peg_name} from a different angle. Image 3 is a cross-sectional image of a {hole_name}. Image 4 is another image of a {hole_name} from a different angle.\nCan the {peg_name} in image 1 and 2 be perfectly inserted into the {hole_name} in image 3 and 4? Please answer with yes or no and the reason.",
                        },
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
                      
            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                return_dict_in_generate=True,  # 出力を辞書形式で返す
                output_scores=True,  # スコアを返す
            )

            # cont.scores の各テンソルに softmax を適用
            probs = [F.softmax(logits, dim=-1) for logits in generated_ids.scores]

            # 最初のトークンの確率分布から最も確率 prob_max とそのトークンID max_token_id を取得
            prob_max, max_token_id = torch.max(probs[0], dim=-1)

            # テンソルから普通の値に変換
            prob_max = np.round(prob_max.item(), 3)
            max_token_id = max_token_id.item()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(
                f"ペグ画像：{image_file_peg}\nホール画像: {image_file_hole}\n応答: {output_text}\nトークンID: {max_token_id}\n確率: {prob_max}\n"
            )
            
            result = np.array([max_token_id, prob_max])

            each_peg_matching_result.append(
                result
            )  # resultをeach_peg_matching_resultに追加
            
        # each_peg_matching_resultをvstackして縦に結合してnumpyの配列にする
        each_peg_matching_result = np.vstack(
            each_peg_matching_result
        )
        
        # 外側のループ（ペグ画像ごとに結果を保存）
        np.save(
            f"/gs/fs/tga-openv/masaruy/general_insertion/Qwen2-VL_Yajima/Matching_result_data/peg_{i+1}_matching_results.npy",
            each_peg_matching_result,
        )
        top_1_data.append(each_peg_matching_result)

    top_1_data = np.vstack(top_1_data)
    
    # 行列の1列目を `response` に、2列目を `probabilities` に代入
    responses = top_1_data[:, 0].astype(int)  # 1列目
    probabilities = top_1_data[:, 1]  # 2列目
    peg_labels = image_files_peg_label
    hole_labels = image_files_hole_label
    
    # Confusion Matrix の描画
    plot_confusion_matrix_with_text2(
        peg_labels,
        hole_labels,
        responses,
        probabilities,
        "Confusion Matrix of Toys Rank",
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
