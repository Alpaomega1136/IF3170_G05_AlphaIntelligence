# IF3170_G05_AlphaIntelligence

Implementasi Hill-Climbing, Simulated Annealing, dan Genetic Algorithm untuk memaksimalkan total value paket dalam kontainer truk, termasuk dukungan multi-truck.

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
| 13524059 | Raymond Jonathan Dwi Putra J. | |