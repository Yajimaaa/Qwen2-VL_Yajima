from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

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

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/peg/1_RJ45_p.png",
                },
                {
                    "type": "image",
                    "image": "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/peg2/1_RJ45_p2.png",
                },
                {
                    "type": "image",
                    "image": "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/hole/1_RJ45_h.png",
                },
                {
                    "type": "image",
                    "image": "/gs/fs/tga-openv/masaruy/general_insertion/object_images/connector_2/hole2/1_RJ45_h.png",
                },
                {
                    "type": "text",
                    #"text": "Explain about image 3."
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
        return_dict_in_generate=True,  # 出力を辞書形式で返す
        output_scores=True,  # スコアを返す
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)


if __name__ == "__main__":
    main()
