import os
import requests
from urllib.parse import urlparse, parse_qs, unquote
from tqdm import tqdm
import time
import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np

source_folder = 'Source_video'
output_folder = 'Out_frames'


def create_source_video_directory():
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)  # Создаем папку для исходных видео


def create_output_frames_directory():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Создаем папку для выходных кадров


def delete_video(video_path):
    if os.path.exists(video_path):
        os.remove(video_path)  # Удаляем ранее скачанное видео


class VideoProcessor:
    """
        Класс VideoProcessor служит для загрузки и обработки видеофайлов, полученных из облачного хранилища.
        Он обеспечивает извлечение кадров из видео при условии нахождения автомобильного номера в зоне интереса.

        Атрибуты:
        ----------
        urls_list : list
            Коллекция URL-адресов исходных видеофайлов
        polygons_dict : dict
            Словарь, содержащий полигоны для каждой камеры, используемые при обработке видео
        confidence : float
            Уровень уверенности для обнаружения объектов (по умолчанию 0.4)
        is_show_bboxes : bool, optional
            Опция, указывающая, следует ли отображать bounding boxes на кадрах (по умолчанию False)
        stride_frame : int, optional
            Шаг обработки кадров, т.е. каждый n-ый кадр будет обработан (по умолчанию 2)
        is_download_videos : bool, optional
            Флаг, указывающий, следует ли скачивать видеофайлы или они уже находятся у вас в папке Source_video (по умолчанию True)
        is_processing : bool, optional
            Флаг, указывающий, следует ли выполнять обработку видео и извлечение кадров (по умолчанию True)
        is_delete_video : bool, optional
            Флаг, указывающий, следует ли удалять исходное видео после обработки (по умолчанию True)
        """

    def __init__(self, urls_list, polygons_dict, confidence=0.4, is_show_bboxes=False, stride_frame=2,
                 is_download_videos=True, is_processing=True, is_delete_video=True):

        self.urls_list = urls_list  # Список ссылок исходных видео
        self.polygons_dict = polygons_dict  # Словарь полигонов по камерам
        self.confidence = confidence  # Уверенность
        self.is_show_bboxes = is_show_bboxes  # Выводить или нет bboxes на кадрах
        self.stride_frame = stride_frame  # Шаг сохранения кадров
        self.is_download_videos = is_download_videos  # Скачивать файлы или они уже лежат в папке Source_video
        self.is_processing = is_processing  # Выполнять обработку видео и извлечение кадров или нет
        self.is_delete_video = is_delete_video  # Следует ли удалять скачанное видео после обработки

    def processing(self, video_source_path, video_source_name):

        video_info = sv.VideoInfo.from_video_path(video_source_path)
        print(f'Длина видео: {video_info.total_frames // video_info.fps} сек')
        print(f'Разрешение: {video_info.width}x{video_info.height}, fps: {video_info.fps} кадров/сек')
        print(f'Шаг считывания кадров из видео: {self.stride_frame}\n')
        print('Идет извлечение подходящих кадров из видео. Ожидайте...')

        video_source_name = os.path.splitext(video_source_name)[0]  # Убираем расширение файла из названия
        camera_id = int(video_source_name.split('.')[3].split('_')[0])  # Номер камеры из названия файла
        polygon = self.polygons_dict.get(camera_id)  # Полигон для конкретной камеры

        if not os.path.exists(f'{output_folder}/{video_source_name}'):
            os.makedirs(f'{output_folder}/{video_source_name}')  # Создаем папку для кадров, если она еще не существует

        model_vehicle = YOLO('Models/yolov8n.pt')  # Модель обнаружения транспорта
        selected_classes = [2, 3, 7, 5]  # классы 'car', 'motorcycle', 'truck', 'bus'

        model_numberplate = YOLO('Models/numberplate_model_v3.pt')  # Модель обнаружения автомобильного номера

        frame_generator = sv.get_video_frames_generator(video_source_path, stride=self.stride_frame)

        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.DEFAULT,
                                                         thickness=5,
                                                         color_lookup=sv.ColorLookup.CLASS)

        bounding_box_annotator_np = sv.BoundingBoxAnnotator(color=sv.Color.YELLOW,
                                                            thickness=5,
                                                            color_lookup=sv.ColorLookup.CLASS)

        polygon_zone = sv.PolygonZone(polygon=polygon,
                                      triggering_anchors=(
                                          [sv.Position.BOTTOM_CENTER, sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT]
                                      ))

        polygon_zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone,
                                                         color=sv.Color(255, 128, 0),
                                                         thickness=5, display_in_zone_count=False)

        saved_frame = 0  # Число сохранённых кадров из видео
        detections_np_last_frame = sv.Detections(xyxy=np.empty((0, 4)))  # Инициализация пустого детекшена

        for frame in tqdm(frame_generator, total=video_info.total_frames // self.stride_frame, colour='green'):

            result = model_vehicle(frame, conf=self.confidence, iou=0.4, imgsz=640,
                                   agnostic_nms=True, verbose=False, )[0]

            detections = sv.Detections.from_ultralytics(result)
            detections = detections[np.isin(detections.class_id, selected_classes)]
            mask = polygon_zone.trigger(detections)
            detections = detections[mask]

            if detections.class_id.size > 0:  # если обнаружены автомобили

                result_np = model_numberplate(frame, conf=0.05, imgsz=640,
                                              verbose=False, )[0]

                detections_np = sv.Detections.from_ultralytics(result_np)
                mask_np = polygon_zone.trigger(detections_np)
                detections_np = detections_np[mask_np]

                if detections_np.class_id.size > 0:  # если обнаружены номера

                    # Определяем сместились ли bboxes номеров по оси Y относительно предыдущего кадра
                    differences = (detections_np.xyxy[:, 3] - detections_np.xyxy[:, 1])[:, np.newaxis]  # смещение по оси Y
                    differences_last_frame = (detections_np_last_frame.xyxy[:, 3] - detections_np_last_frame.xyxy[:, 1])[:, np.newaxis]

                    # Если количество обнаружений предыдущего кадра больше, чем на текущем, убираем лишние обнаружения из предыдущего кадра
                    if len(differences_last_frame) > len(differences):
                        differences_last_frame = differences_last_frame[:len(differences)]

                    # Если количество обнаружений предыдущего кадра меньше, чем на текущем кадре, добавляем в предыдущий кадр нули
                    elif len(differences_last_frame) < len(differences):
                        differences_last_frame = np.pad(differences_last_frame,
                                                        ((0, len(differences) - len(differences_last_frame)), (0, 0)),
                                                        'constant')

                    # Вычисляем разницу между текущим и предыдущим кадрами
                    frame_differences = differences - differences_last_frame

                    detections_np_last_frame = detections_np  # Записываем текущие обнаружения в предыдущий кадр

                    # проверяем, если хоть одна разница между текущим и предыдущим кадрами больше 5 пикселов (т.е. номер движется)
                    if np.any(np.abs(frame_differences) > 5.0):

                        if self.is_show_bboxes:
                            frame = bounding_box_annotator.annotate(frame, detections)
                            frame = bounding_box_annotator_np.annotate(frame, detections_np)
                            frame = polygon_zone_annotator.annotate(frame)

                        saved_frame += 1
                        cv2.imwrite(f'{output_folder}/{video_source_name}/{video_source_name}_{saved_frame:06d}.jpg', frame)

        print('Извлечение завершено!')
        print(f'Сохранено кадров: {saved_frame}')
        print(f'Всего кадров в видео: {video_info.total_frames}')
        print(f'Всего обработано кадров: {video_info.total_frames// self.stride_frame}')

    def run(self):
        """
            Метод run запускает процесс скачивания, обработки видео и извлечения кадров, удаление скачанного видео.
            """

        start_time = time.time()  # Запоминаем текущее время перед началом выполнения кода

        create_source_video_directory()
        create_output_frames_directory()

        if self.is_download_videos:  # Загрузка из облака

            print('-' * 150)
            print(f'КОЛИЧЕСТВО ССЫЛОК ВИДЕОФАЙЛОВ: {len(self.urls_list)}')
            print('-' * 70)

            for index, link in enumerate(self.urls_list):

                query = urlparse(link).query
                params = parse_qs(query)
                video_source_name = unquote(params['files'][0])  # Извлекаем имя файла

                print(f'{index + 1}. Скачивается файл {video_source_name}')

                try:
                    response = requests.get(link, stream=True, timeout=10)  # Скачиваем видео
                    response.raise_for_status()
                except requests.exceptions.RequestException as err:
                    print(f'Ошибка при скачивании файла {video_source_name}: {err}')
                    continue

                total_size = int(response.headers.get('content-length', 0))  # Общий размер файла

                # Скачиваем файл
                with open(os.path.join(source_folder, video_source_name), 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024, unit='KB',
                                      colour='green'):
                        if chunk:
                            f.write(chunk)

                video_path = os.path.join(source_folder, video_source_name)  # Получаем путь к текущему видео файлу

                if self.is_processing:
                    self.processing(video_path, video_source_name)  # Обработка видео, извлечение годных изображений

                if self.is_delete_video:
                    delete_video(video_path)  # Удаляем скачанный видеофайл

                end_time = time.time()  # Запоминаем текущее время после выполнения кода
                print(f'Время полного цикла обработки видео: {round(end_time - start_time)} сек')  # Время выполнения кода
                print('-' * 70)

        else:  # Берем из папки Sourse_video

            print('-' * 150)
            print(f"КОЛИЧЕСТВО ВИДЕОФАЙЛОВ В ПАПКЕ {source_folder}: {len([f for f in os.listdir(source_folder) if f.endswith('.mp4')])}")
            print('-' * 70)

            index_frame = 0

            for file in os.listdir(source_folder):
                if file.endswith('.mp4'):
                    index_frame += 1
                    video_source_name = file

                    print(f'{index_frame}. Взят на обработку файл {video_source_name}')

                    video_path = os.path.join(source_folder, video_source_name)  # Получаем путь к текущему видео файл

                    if self.is_processing:
                        self.processing(video_path, video_source_name)  # Обработка видео, извлечение годных изображений

                    if self.is_delete_video:
                        delete_video(video_path)  # Удаляем скачанный видеофайл

                    end_time = time.time()  # Запоминаем текущее время после выполнения кода
                    print(f'Время полного цикла обработки видео: {round(end_time - start_time)} сек')  # Время выполнения кода
                    print('-' * 70)

        print('РАБОТА ПРОГРАММЫ ЗАВЕРШЕНА!!!')
