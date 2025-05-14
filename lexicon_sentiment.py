import re

positive_words = [
    "sangat", "membantu", "bagus", "mantap", "good", "baik", "promo",
    "top", "mudah", "cepat", "suka", "pelayanan", "ramah", "terimakasih",
    "terima", "semoga", "keren", "tepat", "puas", "nyaman", "layanan",
    "memuaskan", "aman", "memudahkan", "murah", "sesuai", "sukses",
    "mantab", "nice", "bermanfaat", "lancar", "senang", "terbaik",
    "mantul", "alhamdulillah", "enak", "bener", "terbantu", "makasih",
    "mempermudah", "berguna", "praktis", "menyenangkan", "gampang",
    "jos", "cepet", "hemat", "proses", "bantu", "maju"
]
negative_words = [
    "tidak", "ga", "gak", "lama", "tolong", "kenapa", "malah", "sekali",
    "gk", "mahal", "susah", "masuk", "apk", "saldo", "update", "biaya",
    "kecewa", "kok", "masih", "parah", "tdk", "ribet", "gabisa", "cancel",
    "minta", "terlalu", "kurang", "perbaiki", "masalah", "salah", "gimana",
    "sampah", "kecewa", "sedih", "marah", "hilang", "jelek", "error", "blokir", 
    "eror", "payah", "telat", "suruh", "alasan", "hapus",
    "males", "limit", "mengecewakan", "ko", "lemot", "gajelas",
    "diperbaiki", "kendala", "bug", "kasian", "denda", "ilang",
    "buruk", "gagal", "duit", "batal"
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
    "pihak", "biar", "jd", "seperti", "coba", "lokasi"
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
