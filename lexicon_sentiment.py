import re

positive_words = [
    "sangat", "banget", "membantu", "mbantu", "bagus", "mantap", "good", "baik", "promo",
    "top", "mudah", "gampang", "cepat", "cepet", "suka", "sukak", "pelayanan", "ramah", "terimakasih",
    "terima", "keren", "tepat","pas", "puas", "seneng", "nyaman", "aman", "layanan",
    "memuaskan", "memudahkan", "murah", "sesuai", "sukses",
    "mantab", "mantap", "nice", "bermanfaat", "lancar", "senang", "terbaik",
    "mantul", "alhamdulillah", "enak", "bener", "benar", "terbantu", "makasih",
    "mempermudah", "berguna", "praktis", "menyenangkan", "gampang",
    "jos", "hemat", "proses", "bantu", "maju", "cinta"
]
negative_words = [
    "tidak", "ga", "gak", "lama","lambat", "tolong", "kenapa", "malah", "sekali",
    "gk", "mahal", "susah", "sulit","masuk", "apk", "saldo", "update", "biaya",
    "kecewa", "kok", "masih", "parah", "tdk", "ribet", "gabisa", "cancel",
    "minta", "terlalu", "kurang", "perbaiki", "masalah", "salah", "gimana",
    "sampah", "kecewa", "sedih", "marah", "bingung", "hilang", "jelek","jele", "error", "blokir", 
    "eror", "payah", "telat", "suruh", "alasan", "hapus",
    "males", "limit", "mengecewakan", "ko", "lemot", "gajelas",
    "diperbaiki", "kendala", "bug", "kasian", "denda", "ilang", "lag",
    "buruk", "gagal", "duit", "batal", "hilang", "gak sesuai", "tidak sesuai",
    "ga sesuai", "jebakan", "tidak suka", "aneh"
]
neutral_words = [
    "padahal", "ya", "banyak", "udah", "buat", "jadi",
    "lebih", "pake", "sama", "selalu", "gofood", "kasih", "terus",
    "sekarang", "juga", "dari", "dengan", "pakai", "akun", "sering", "pesan",
    "drivernya", "itu", "apa", "bayar", "baru", "go", "kalau", "dapat",
    "lah", "order", "harus", "sih", "saat", "dulu", "karena", "hari",
    "aplikasinya", "dong", "pas", "bintang", "waktu", "kali",
    "pernah", "semua", "sy", "lain", "sampai", "pesen", "jauh", "atau",
    "aku", "dapet", "diskon", "jelas", "ongkir", "tp", "harga",
    "dalam", "semakin", "cuma", "nunggu", "d", "klo", "mohon", "belum",
    "transaksi", "kita", "beli", "jam", "saja", "buka", "gopaylater",
    "pesanan", "jangan", "pada", "hp", "up", "gocar", "tiba", "kan",
    "mana", "pembayaran", "masa", "setiap", "uang", "orang", "naik", "lg",
    "dah", "resto", "customer", "cs", "upgrade", "food", "fitur", "g",
    "setelah", "langsung", "orderan", "sampe", "ganti", "sendiri",
    "pihak", "biar", "jd", "seperti", "coba", "lokasi", "semoga"
]


inset_lexicon = {}

for word in positive_words:
    inset_lexicon[word] = 1
for word in negative_words:
    inset_lexicon[word] = -1
for word in neutral_words:
    inset_lexicon[word] = 0

def get_sentiment(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)

    score = 0
    for token in tokens:
        score += inset_lexicon.get(token, 0)

    if score > 0:
        return 'positif'
    elif score < 0:
        return 'negatif'
    else:
        return 'netral'
