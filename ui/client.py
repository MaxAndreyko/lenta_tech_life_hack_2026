import gradio as gr
import pandas as pd

from ui import util
from ui.util import load_css


class UIClient:
    blocks: gr.Blocks
    process_video_with_progress_callback = None

    def __init__(self, process_video_with_progress_callback):
        self.process_video_with_progress_callback = process_video_with_progress_callback
        with gr.Blocks(title="Распознавание ценников") as blocks:
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
                fn=util.get_preview_video_frame,
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
                fn=self.process_video_with_progress_callback,
                inputs=[video_input, frame_interval, confidence_threshold],
                outputs=[data_table, status_output, download_output]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[download_output]
            )

            # Автопредпросмотр при загрузке
            video_input.change(
                fn=util.get_preview_video_frame,
                inputs=[video_input],
                outputs=[preview_image]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[preview_image]
            )

        self.blocks = blocks

    def launch(self, server_name: str, server_port: int, share: bool, show_error: bool):
        self.blocks.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
            ),
            css=load_css("ui/index.css"),
            show_error=show_error
        )