# IF3170_G05_AlphaIntelligence

## Deskripsi

Repository ini berisi implementasi Tugas Besar 1 IF3170 Inteligensi Artifisial. Permasalahan yang diselesaikan adalah memaksimalkan total value paket yang dapat dimuat ke dalam peti kemas truk. Setiap solusi harus memenuhi batas dimensi kontainer, tidak memiliki paket yang saling bertumpang tindih atau melayang, tidak menggunakan paket fragile sebagai penopang, dan tidak melebihi kapasitas berat truk.

Program menyediakan tiga algoritma local search, yaitu Hill-Climbing, Simulated Annealing, dan Genetic Algorithm. Hill-Climbing mendukung varian Steepest-Ascent, Stochastic, Sideways Move, dan Random Restart. Input program menggunakan JSON, sedangkan hasil setiap eksperimen disimpan dalam folder `output/` sebagai ringkasan, state awal dan akhir, riwayat objective, serta visualisasi.

## Struktur Repository

```text
.
|-- src/
|   |-- core.py                 # Representasi state, constraint, objective, dan neighbor
|   |-- hill_climbing.py        # Semua varian Hill-Climbing
|   |-- simulated_annealing.py  # Algoritma Simulated Annealing
|   |-- genetic_algorithm.py    # Algoritma Genetic Algorithm
|   |-- parser.py               # Pembacaan dan validasi input JSON
|   |-- output_manager.py       # Penyimpanan ringkasan dan hasil eksperimen
|   |-- visualization.py        # Visualisasi state dan riwayat objective
|   `-- main.py                 # Entry point program
|-- test/
|   |-- hill_climbing/          # Testcase JSON Hill-Climbing
|   |-- simulated_annealing/    # Testcase JSON Simulated Annealing
|   `-- genetic_algorithm/      # Testcase JSON Genetic Algorithm
|-- output/                     # Hasil eksperimen yang dihasilkan program
|-- docs/                       # Dokumen dan bahan laporan
`-- README.md
```

## Setup

Gunakan Python 3.10 atau lebih baru, lalu pasang dependensi:

```bash
python -m pip install matplotlib
```

## Menjalankan Program

Perintah untuk Hill-Climbing, Simulated Annealing, dan Genetic Algorithm sama. Jalankan dari direktori utama repository:

```text
python -m src.main <file_TC.json>
```

Ganti `<file_TC.json>` dengan path file testcase JSON. Contoh:

```bash
python -m src.main test/genetic_algorithm/population_20_iterations_100/run_1.json
```

Algoritma dipilih melalui `algorithm.name` dan parameternya melalui `algorithm.parameters` dalam file input. Hasil ditampilkan pada terminal dan disimpan di folder `output/`.

## Pembagian Tugas

| NIM | Nama | Tugas |
| --- | --- | --- |
| 13524006 | Gabriella Botimada Lubis | Implementasi algoritma Simulated Annealing;<br>Membuat visualisasi hasil;<br>Mengerjakan laporan |
| 13524038 | An Dafa Anza Avansyah | Implementasi Genetic Algorithm;<br>Implementasi bonus *multi-truck*;<br>Mengerjakan laporan |
| 13524059 | Raymond Jonathan Dwi Putra J. |Implementasi algoritma Hill-Climbing semua varian;<br> Membuat pondasi program;<br>Mengerjakan laporan|
