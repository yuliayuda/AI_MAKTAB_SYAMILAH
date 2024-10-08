project-root/
│
├── data/               # Dataset dan file terkait
│   ├── raw/            # Data mentah
│   │   ├── prepare_data.py
│   ├── processed/      # Data yang sudah diolah
│   ├── preprocessing_data.py
│
├── models/             # Direktori untuk berbagai model
│   ├── BERT.py
│   ├── LSTM.py
│   ├── Transformer.py
│
├── modules/            # Berisi modul-modul khusus
│   ├── qa_module.py    
│   ├── retrieval_module.py
│   ├── translation_module.py
│   ├── summarization_module.py
│
├── utils/              # Fungsi-fungsi utilitas
│   ├── preprocessing.py
│   ├── postprocessing.py
│
├── config/             # Konfigurasi untuk model dan pipeline
│   └── config.py
│
├── training/           # Modul untuk melatih model
│   └── train_model.py
│
├── evaluation/         # Modul untuk evaluasi model
│   └── evaluate_model.py
│
├── logs/               # Log untuk debugging
│   └── logging.py
│
├── main.py             # Script utama untuk menjalankan sistem
│
└── README.md           # Dokumentasi proyek
