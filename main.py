import os
from dataset_generator import DatasetGenerator
from tinyLama import Trainer,ModelManager,Utils,TextGenerator



if __name__ == "__main__":
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    save_path = "tinyllama-tuned"
    data_path = "./datasets/json/dataset.json"
    book_path = "./datasets/pdf/a.s.-pushkin-sbornik-skazok-dlya-detey.pdf"

    dataset_generator = DatasetGenerator()
    dataset_path = dataset_generator.create_json_dataset_1book(book_path);
    # dataset_generator.create_json_dataset_all_books()


    model_manager = ModelManager(model_id, save_path)
    trainer = Trainer(model_manager)

    # Train the model
    trainer.train_model(dataset_path)

    # Fine-tune the model
    last_checkpoint = Utils.get_last_checkpoint(save_path)
    if last_checkpoint:
        trainer.fine_tune_model(data_path, last_checkpoint)

    # Generate text
    generator = TextGenerator(save_path)
    print(generator.generate_text("Once upon a time", max_length=100))
