import os
import threading
import tempfile

import gradio as gr
import pandas as pd
import uvicorn

from backend.app.Server import Server
from backend.app.processor.PriceTagProcessor import PriceTagProcessor
from ui.client import UIClient


class PriceTagApp:
    """Main class for price tag recognition app"""

    def __init__(self):
        self.ui_client = None
        self.processor = None
        self.server = None
        self._api_thread = None


    def _init_processor(self):
        if self.processor is None:
            print("⏳ Загрузка моделей...")
            self.processor = PriceTagProcessor()
            print("✅ Модели загружены!")
        return self.processor


    def process_video_with_progress(
        self,
        video_path,
        frame_interval,
        confidence_threshold,
        progress=gr.Progress()
    ):
        if video_path is None:
            return None, "### ❌ Загрузите видео", None, None

        try:
            processor = self._init_processor()

            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                output_path = tmp.name

            progress(0.1, desc="🔍 Анализируем видео...")

            results = processor.process_video(
                video_path=video_path,
                output_csv=output_path,
                frame_interval=int(frame_interval),
                confidence_threshold=float(confidence_threshold)
            )

            progress(0.9, desc="📊 Формируем отчёт...")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                df = pd.read_csv(output_path)
                total_tags = len(df)

                avg_price = df['price_regular'].mean() if total_tags > 0 else 0
                avg_conf = df['confidence'].mean() * 100 if total_tags > 0 else 0
                categories = df['category'].nunique() if total_tags > 0 else 0

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
                return None, "### ⚠️ Ценники не найдены", None

        except Exception as e:
            error_msg = f"""
            ### ❌ Ошибка обработки
            **Тип:** {type(e).__name__}
            **Описание:** {str(e)}
            """
            return None, error_msg, None


    def _init_server(self, upload_dir="uploads", output_dir="output"):
        if self.server is None:
            print("\n🚀 Инициализация API сервера...")
            self.server = Server(upload_dir=upload_dir, output_dir=output_dir)
            print("✅ Сервер готов")
        return self.server


    def _start_api_server(self, host="0.0.0.0", port=8000):
        server = self._init_server()
        app = server.get_app()

        self._api_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": host, "port": port, "log_level": "info"},
            daemon=True
        )
        self._api_thread.start()

        print(f"📡 API: http://localhost:{port}")
        print(f"📚 Docs: http://localhost:{port}/docs")


    def create_ui_client(self):
        print("\n🌐 Initializing UI...")
        return UIClient(self.process_video_with_progress)


    def launch(self, server_name="0.0.0.0", server_port=7860, api_port=8000):
        self._start_api_server(host=server_name, port=api_port)
        self.ui_client = self.create_ui_client()
        self.ui_client.launch(
            server_name=server_name,
            server_port=server_port,
            share=True,
            show_error=True
        )
        print(f"\n✅ UI:  http://localhost:{server_port}")
        print(f"✅ API: http://localhost:{api_port}")