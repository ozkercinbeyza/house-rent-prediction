Bir emlak platformu, farklı şehirlerdeki ev ilanlarını analiz 
ederek kullanıcıların kiralık ev tercihlerine uygun öneriler 
sunmak ve konut kira piyasasını daha doğru tahmin edebilmek 
istemektedir. Bu doğrultuda, evlerin fiziksel özellikleri (oda 
sayısı, büyüklük, banyo sayısı), konumu (şehir, semt), donanım 
durumu (eşyalı/eşyasız) ve kiracı tercihi gibi birçok faktörün kira 
fiyatı üzerindeki etkisi analiz edilmek istenmektedir. 
Bu analizde kullanılan veri seti, Hindistan’daki kiralık ev 
ilanlarından oluşmaktadır. Modelleme süreciyle birlikte kira 
fiyatlarını etkileyen temel faktörler belirlenerek: 
● Kullanıcılara daha isabetli fiyat tahmini yapılması, 
● Ev sahiplerine adil fiyat belirleme konusunda destek 
sağlanması, 
● Yeni kullanıcılar için kişiselleştirilmiş öneri sistemleri 
geliştirilmesi hedeflenmektedir. 
İlk olarak Kaggle platformundan House Rent verisetini Colaba yüklendi.
House_Rent_Dataset.csv dosyasında veriseti sütunları yer almaktadır.
House_Rent_Dataset.py dosyasında Colab üzerinde yazmış olduğum kodlar bulunmaktadır.Burada ne yaptım?
-İlk olarak veri setimde kaç satır , kaç sütun var. Değişken tipleri neler, verisetindeki ilk beş sütun neler,eksik veri var mı gibi sorularla veri setini anlamaya çalıştım.
-Veri seti temizdi eksik veri bulunmuyordu.
-Daha sonra görselleştirme yaparak uygun kodlarla grafikler oluşturdum. Şehir Bazında Ev Sayıları Dağılımı, Mobilya Durumuna Göre Ev Sayıları Dağılımı,Şehirlere Göre Ortalama Kira Dağılımı,90 m² ve Üzeri Evler İçin Ortalama Kira Dağılımı,
Oda Sayısına Göre Ortalama Kira Dağılımı
-Aykırı değerler filtrelendi:Sadece Size değeri 30m² ile500m²
 arasında olan evler seçildi.
-Kategorik veriler One-Hot Encoding ile sayısal hale getirildi.
-Sayısal veriler StandardScaler ile ölçeklendi.
-Hedef değişken olan kira tutarına log dönüşümü uygulandı
 böylece dağılım normalleştirildi.
-Linear Regression,SVR,Random Forest,Decision Tree Modellerinin başarı oranları kaşılaştırıldı.
--Kullandığım modeller arasından en düşük Ortalama Kare Hata(MSE) ve en yüksek R-Kare (başarı oranını) skoru veren Random Forest modeli oldu.
-Modelin öğrenmesi sırasında eğitim ve test verisi %80-%20
 oranında bölündü.
-Hiperparametre optimizasyonu yapılarak model optimize edildi.Bu sayede modelin başarısı %86'nın üzerinde doğruluk sağladı.
-app.py dosyasında yer alan kodlar ile;
-Kullanıcıdan gelecek olan bilgilerle kira tahminini gösteren Streamlit arayüzü tasarlandı.
-Arayüzde kullanıcıdan,
Oda Sayısı (BHK)
Alan (metrekare)
Banyo Sayısı
Şehir
Eşya Durumu
Alan Tipi  bilgilerinin girilmesi istendi.
- Kullanıcının girdiği veriler:
DataFrame'e dönüştürülür,
One-hot encoding uygulanır,
Eksik kolonlar tamamlanır,
Scaler ile normalize edilir,
Model ile tahmin yapılır.
-Yapılan tahmin geçmişe eklenir ve kullanıcıya gösterilir.
-Kullanıcının önceki tahminleri tablo halinde gösterilir.
Bu proje Yapay Zeka dersi için yapılmış olan bir final ödevidir.




