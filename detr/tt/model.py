import os
from typing import List, Dict, Optional
import sys
sys.path.insert(0, '/home/zhongbing/Projects/MLE/table-transformer/detr/infer')
from PIL import Image
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from inference import inference_for_table_recognition
from label_studio_ml.model import LabelStudioMLBase




def convert_bbox_to_labelstudio(bbox, label, result_id, original_width, original_height):
    x_min, y_min, x_max, y_max = bbox
    x = x_min / original_width * 100
    y = y_min / original_height * 100
    width = (x_max - x_min) / original_width * 100
    height = (y_max - y_min) / original_height * 100
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": original_width,
        "original_height": original_height,
        "image_rotation": 0,
        "value": {
            "rotation": 0,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rectanglelabels": [label]
        }
    }


def get_image_size(image_path):
    img = Image.open(image_path)
    width, height = img.size
    return width, height


# 转换数据
def consolidate(data, width, height):
    print(data)
    converted_results = [
        convert_bbox_to_labelstudio(data[i]['bbox'], data[i]['label'], f"result{i + 1}", width, height)
        for i in range(len(data))]

    return [{
        "model_version": "one",
        "score": 0.5,
        "result": converted_results
    }]


def inference(path):
    result = inference_for_table_recognition(path)
    width, height = get_image_size(path)
    return consolidate(result, width, height)


class NewModel(LabelStudioMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        image_path = tasks[0]['data']['image']
        # image_url = get_image_url
        # image_path = self.get_local_path(image_url)
        file_name = image_path.split('/')[-1]
        image_path = "/home/zhongbing/Projects/MLE/LabelStudio/media/upload/3/" + file_name
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return inference(image_path)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')


# if __name__ == '__main__':
#     # result = inference_for_table_recognition("/home/zhongbing/Projects/MLE/table-transformer/detr/img/complex.jpg")
#     print(result)
