import os
import tempfile

import cv2
import gradio as gr
import pandas as pd

from backend.app.processor.PriceTagProcessor import PriceTagProcessor

CUSTOM_CSS = """
.header-container {
    text-align: center;
    margin-bottom: 2rem;
}
.stats-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.upload-box {
    border: 2px dashed #667eea;
    border-radius: 15px;
    padding: 2rem;
    background: #f8f9fa;
}
.footer {
    text-align: center;
    color: #666;
    margin-top: 2rem;
}
"""

class PriceTagApp:
    """Главный класс веб-приложения для распознавания ценников"""

    def __init__(self):
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

    def get_preview_video_frame(self, video_path):
        """Извлечение случайного кадра из видео для предпросмотра"""
        if video_path is None:
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Берем кадр из середины видео
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            return None
        except:
            return None

    def init_ui(self):
        """Создание Gradio интерфейса для версии 6.0+"""

        # В Blocks() больше НЕ передаем theme и css
        with gr.Blocks(title="Распознавание ценников") as demo:
            # Заголовок
            gr.Markdown("""
            <div class="header-container">
                <h1>🏷️ Система автоматического распознавания ценников</h1>
                <p style="font-size: 1.1em;">Загрузите видео со стеллажами и получите структурированный CSV с ценами и скидками</p>
            </div>
            """)

            with gr.Row():
                # Левая колонка
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="upload-box"):
                        gr.Markdown("### 📤 Загрузка видео")

                        video_input = gr.Video(
                            label="Выберите видеофайл",
                            format="mp4",
                            height=300
                        )

                        with gr.Row():
                            preview_btn = gr.Button("👁️ Предпросмотр", size="sm")
                            clear_btn = gr.Button("🗑️ Очистить", size="sm")

                        preview_image = gr.Image(
                            label="Случайный кадр из видео",
                            visible=False,
                            height=200
                        )

                    with gr.Accordion("⚙️ Настройки обработки", open=False):
                        frame_interval = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=5,
                            step=1,
                            label="🎞️ Интервал кадров",
                            info="Меньше = точнее, но медленнее (рекомендуется 5-10)"
                        )

                        confidence_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="🎯 Порог уверенности детекции",
                            info="Выше = строже отбор (для сложного видео уменьшите)"
                        )

                    process_btn = gr.Button(
                        "🚀 Обработать видео",
                        variant="primary",
                        size="lg"
                    )

                # Правая колонка
                with gr.Column(scale=1):
                    status_output = gr.Markdown("""
                    ### ⏳ Ожидание загрузки видео

                    Загрузите видео в левой панели и нажмите **"Обработать"**
                    """)

                    data_table = gr.Dataframe(
                        label="📊 Распознанные ценники",
                        interactive=False,
                        wrap=True
                    )

                    download_output = gr.DownloadButton(
                        label="📥 Скачать CSV с результатами",
                        variant="secondary",
                        visible=False
                    )

            # Подвал
            gr.Markdown("""
            <div class="footer">
            <hr>
            <h3>📋 Формат выходных данных (CSV)</h3>
            </div>
            """)

            gr.Dataframe(
                value=pd.DataFrame({
                    "Поле": ["barcode", "product_name", "price_regular", "price_discount",
                             "discount_percent", "category", "weight_volume", "confidence"],
                    "Описание": ["Штрихкод товара", "Название продукта", "Обычная цена",
                                 "Цена со скидкой", "Процент скидки", "Категория товара",
                                 "Вес/объем", "Уверенность распознавания"],
                    "Пример": ["4601234567890", "Молоко 3.2%", "89.90", "59.90", "33",
                               "Молочная продукция", "930 мл", "0.89"]
                }),
                interactive=False,
                label="Описание полей"
            )

            # Обработчики событий

            # Предпросмотр кадра
            preview_btn.click(
                fn=self.get_preview_video_frame,
                inputs=[video_input],
                outputs=[preview_image]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[preview_image]
            )

            # Очистка
            def clear_all():
                return None, None, None, None, None

            clear_btn.click(
                fn=clear_all,
                outputs=[video_input, preview_image, data_table,
                         status_output, download_output]
            ).then(
                fn=lambda: gr.update(visible=False),
                outputs=[preview_image, download_output]
            )

            # Обработка видео
            process_btn.click(
                fn=self.process_video_with_progress,
                inputs=[video_input, frame_interval, confidence_threshold],
                outputs=[data_table, status_output, download_output]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[download_output]
            )

            # Автопредпросмотр при загрузке
            video_input.change(
                fn=self.get_preview_video_frame,
                inputs=[video_input],
                outputs=[preview_image]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[preview_image]
            )

        return demo
