import os
import tempfile

import gradio as gr
import pandas as pd

from backend.app.processor.PriceTagProcessor import PriceTagProcessor
from ui.client import UIClient


class PriceTagApp:
    """Main class for price tag recognition app"""

    def __init__(self):
        self.ui_client = None
        self.processor = None
        self.current_results = None

    def _init_processor(self):
        """Ленивая инициализация процессора (загружаем только когда нужно)"""
        if self.processor is None:
            print("⏳ Загрузка моделей...")
            self.processor = PriceTagProcessor(
                detector_model="yolov8n.pt",
            )
            print("✅ Модели загружены!")
        return self.processor

    def process_video_with_progress(self, video_path, frame_interval,
                                    confidence_threshold, progress=gr.Progress()):
        """Обработка видео с отображением прогресса"""

        if video_path is None:
            return None, "### ❌ Пожалуйста, загрузите видео", None, None

        try:
            processor = self._init_processor()

            # Создаем временный файл для результатов
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                output_path = tmp.name

            progress(0.1, desc="🔍 Анализируем видео...")

            # Запускаем обработку
            results = processor.process_video(
                video_path=video_path,
                output_csv=output_path,
                frame_interval=int(frame_interval),
                confidence_threshold=float(confidence_threshold)
            )

            progress(0.9, desc="📊 Формируем отчёт...")

            # Читаем результаты
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                df = pd.read_csv(output_path)
                total_tags = len(df)

                if total_tags > 0:
                    avg_price = df['price_regular'].mean()
                    avg_conf = df['confidence'].mean() * 100
                    categories = df['category'].nunique()
                else:
                    avg_price = 0
                    avg_conf = 0
                    categories = 0

                stats = f"""
                ### ✅ Обработка завершена

                - 🏷️ **Найдено ценников:** {total_tags} шт.
                - 💰 **Средняя цена:** {avg_price:.2f} руб.
                - 🎯 **Средняя уверенность:** {avg_conf:.1f}%
                - 🏪 **Категорий товаров:** {categories}
                """

                progress(1.0, desc="✅ Готово!")
                return df, stats, output_path
            else:
                return None, "### ⚠️ Ценники не найдены на видео", None

        except Exception as e:
            error_msg = f"""
            ### ❌ Ошибка обработки

            **Тип:** {type(e).__name__}
            **Описание:** {str(e)}

            Проверьте:
            - Формат видео (mp4, avi, mov)
            - Качество видео
            - Наличие ценников в кадре
            """
            return None, error_msg, None

    def create_ui_client(self) -> UIClient:
        print("\n🌐 Initializing UI...")
        return UIClient(self.process_video_with_progress)

    def launch(self, server_name, server_port):
        self.ui_client = self.create_ui_client()

        self.ui_client.launch(
            server_name=server_name,
            server_port=server_port,
            share=True,
            show_error=True
        )
        print(f"Service is available at: http://localhost:{server_port}")