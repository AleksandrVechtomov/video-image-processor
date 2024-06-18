from core import VideoProcessor
import numpy as np


# Ссылки на видео файлы из облака
urls = [
    'https://cloud.kscgroup.ru/index.php/s/yagZNcqdSqM8CZc/download?path=%2F21.05&files=10.121.15.248_01_20240521060836261_1.mp4',
]

# Полигоны для каждой камеры
polygons = {
    247: np.array([[1273, 1410], [2565, 694], [2385, 602], [2077, 282], [501, 794], [905, 1058], [1185, 1282]]),
    248: np.array([[1061, 642], [2305, 734], [2681, 1154], [2681, 1250], [245, 1222]]),
    249: np.array([[325, 618], [1953, 1222], [2201, 630], [2213, 378], [693, 218], [489, 466]]),
    252: np.array([[485, 914], [1549, 1510], [2105, 1506], [2273, 1266], [2573, 990], [1129, 610], [921, 750]])
}


process = VideoProcessor(urls_list=urls,
                         polygons_dict=polygons,
                         confidence=0.5,
                         is_show_bboxes=False,
                         stride_frame=10,
                         is_download_videos=True,
                         is_processing=True,
                         is_delete_video=False)
process.run()
