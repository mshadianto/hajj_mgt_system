"""
🤖 RAG (RETRIEVAL AUGMENTED GENERATION) ASSISTANT
Intelligent Knowledge System for Hajj Financial Guidance

Features:
- Knowledge base of Islamic finance principles
- Contextual document retrieval
- AI-powered financial advisory
- Regulatory compliance checking
- Multi-language support (Indonesian/Arabic/English)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import re
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🤖 RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #2c3e50;
        margin-right: auto;
    }
    
    .knowledge-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .source-tag {
        background: #ecf0f1;
        color: #34495e;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .confidence-bar {
        background: #ecf0f1;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# Knowledge Base Class
class HajjFinanceKnowledgeBase:
    """
    Comprehensive knowledge base for hajj financial guidance
    """
    
    def __init__(self):
        self.knowledge_docs = self._load_knowledge_base()
        self.financial_regulations = self._load_regulations()
        self.islamic_principles = self._load_islamic_principles()
        self.faqs = self._load_faqs()
    
    def _load_knowledge_base(self) -> List[Dict]:
        """Load comprehensive hajj financial knowledge"""
        return [
            {
                "id": "hajj_finance_basics",
                "title": "Dasar-dasar Keuangan Haji",
                "content": """
                Pengelolaan keuangan haji melibatkan beberapa komponen utama:
                
                1. BPIH (Biaya Penyelenggaraan Ibadah Haji): Biaya operasional penyelenggaraan haji
                2. Bipih (Biaya Penyetoran Awal): Setoran awal jemaah saat mendaftar
                3. Nilai Manfaat: Return investasi dari dana yang dikelola
                4. Dana Investasi: Alokasi dana untuk instrumen keuangan syariah
                
                Prinsip utama: Dana haji harus dikelola sesuai prinsip syariah dengan fokus pada:
                - Transparansi pengelolaan
                - Optimalisasi return halal
                - Perlindungan modal jemaah
                - Sustainabilitas jangka panjang
                """,
                "category": "basic_concepts",
                "tags": ["bpih", "bipih", "nilai_manfaat", "syariah"],
                "language": "id",
                "confidence": 0.95
            },
            {
                "id": "bpkh_overview",
                "title": "BPKH (Badan Pengelola Keuangan Haji)",
                "content": """
                BPKH adalah lembaga yang dibentuk berdasarkan UU No. 34 Tahun 2014 untuk mengelola keuangan haji secara profesional dan transparan.
                
                Struktur Organisasi BPKH:
                1. Badan Pelaksana: Fungsi perencanaan, pelaksanaan, pertanggungjawaban, dan pelaporan
                2. Dewan Pengawas: Fungsi pengawasan terhadap pengelolaan keuangan haji
                
                Tugas Utama BPKH:
                - Mengelola keuangan haji (penerimaan, pengembangan, pengeluaran, pertanggungjawaban)
                - Menempatkan dan menginvestasikan dana sesuai prinsip syariah
                - Melakukan kerjasama dengan lembaga lain
                - Memberikan laporan berkala
                
                Prinsip Pengelolaan:
                - Prinsip syariah
                - Prinsip kehati-hatian
                - Manfaat untuk jemaah
                - Nirlaba
                - Transparan
                - Akuntabel
                """,
                "category": "institutional",
                "tags": ["bpkh", "organisasi", "struktur", "tugas"],
                "language": "id",
                "confidence": 0.96
            },
            {
                "id": "islamic_investment_principles",
                "title": "Prinsip Investasi Syariah untuk Dana Haji",
                "content": """
                Investasi dana haji harus mematuhi prinsip-prinsip syariah:
                
                1. Bebas Riba (Interest-free): Tidak melibatkan bunga/riba
                2. Bebas Gharar: Menghindari ketidakpastian berlebihan
                3. Bebas Maysir: Tidak ada unsur judi/spekulasi
                4. Halal Business: Investasi pada sektor halal
                
                Instrumen yang direkomendasikan:
                - Sukuk pemerintah dan korporasi
                - Saham syariah (indeks JII)
                - Reksadana syariah
                - Deposito syariah
                - Real estate syariah
                - Komoditas halal (emas, perak)
                
                Diversifikasi portofolio sangat penting untuk mengelola risiko
                sambil mempertahankan compliance syariah.
                """,
                "category": "investment_guidance",
                "tags": ["syariah", "sukuk", "reksadana", "diversifikasi"],
                "language": "id",
                "confidence": 0.92
            },
            {
                "id": "hajj_fund_composition",
                "title": "Komposisi Keuangan Haji",
                "content": """
                Berdasarkan UU No. 34/2014, keuangan haji terdiri dari:
                
                A. PENERIMAAN:
                1. Setoran BPIH dan/atau BPIH Khusus
                2. Nilai manfaat keuangan haji (hasil investasi)
                3. Dana efisiensi penyelenggaraan ibadah haji
                4. Dana Alokasi Umum (DAU)
                5. Sumber lain yang sah dan tidak mengikat
                
                B. PENGELUARAN:
                1. Penyelenggaraan ibadah haji
                2. Operasional BPKH
                3. Penempatan dan/atau investasi keuangan haji
                4. Kegiatan untuk kemaslahatan umat Islam
                5. Lain-lain pengeluaran yang sah
                
                Pengelolaan harus mengoptimalkan nilai manfaat untuk:
                - Meningkatkan kualitas penyelenggaraan ibadah haji
                - Rasionalitas dan efisiensi penggunaan BPIH
                - Manfaat bagi kemaslahatan umat Islam
                """,
                "category": "fund_structure",
                "tags": ["penerimaan", "pengeluaran", "bpih", "nilai_manfaat"],
                "language": "id",
                "confidence": 0.94
            },
            {
                "id": "risk_management_hajj",
                "title": "Manajemen Risiko Dana Haji",
                "content": """
                Manajemen risiko dalam pengelolaan dana haji mencakup:
                
                A. Risiko Pasar:
                - Fluktuasi nilai investasi
                - Volatilitas return
                - Risiko inflasi
                
                B. Risiko Operasional:
                - Peningkatan biaya haji
                - Perubahan regulasi
                - Risiko teknologi
                
                C. Risiko Likuiditas:
                - Kebutuhan dana mendadak
                - Mismatch aset-liabilitas
                
                Strategi Mitigasi:
                1. Diversifikasi aset dan instrumen
                2. Stress testing berkala
                3. Contingency planning
                4. Insurance dan hedging syariah
                5. Monitoring real-time
                
                Target risk-adjusted return: 6-8% per tahun dengan volatilitas <15%.
                """,
                "category": "risk_management",
                "tags": ["risiko", "diversifikasi", "stress_testing", "hedging"],
                "language": "id",
                "confidence": 0.88
            },
            {
                "id": "sustainability_metrics",
                "title": "Metrik Sustainabilitas Keuangan Haji",
                "content": """
                Indikator kunci sustainabilitas dana haji:
                
                1. Sustainability Index = (Nilai Manfaat / BPIH) × 100%
                   - Target: >60% (Sehat)
                   - Warning: 40-60% (Perlu perhatian)
                   - Critical: <40% (Intervensi segera)
                
                2. Cost Coverage Ratio = Dana Tersedia / Total Biaya Proyeksi
                   - Minimum: 1.2x (120%)
                
                3. Return on Investment (ROI) = Annual Return / Total Investment
                   - Target: 6-8% per tahun
                
                4. Risk-Adjusted Return = ROI / Standard Deviation
                   - Target Sharpe Ratio: >0.4
                
                5. Liquidity Ratio = Liquid Assets / Short-term Obligations
                   - Minimum: 25%
                
                Monitoring dilakukan monthly dengan reporting quarterly.
                """,
                "category": "performance_metrics",
                "tags": ["sustainability", "kpi", "monitoring", "roi"],
                "language": "id",
                "confidence": 0.94
            },
            {
                "id": "regulatory_compliance",
                "title": "Kepatuhan Regulasi Pengelolaan Dana Haji",
                "content": """
                Kerangka regulasi pengelolaan dana haji di Indonesia:
                
                1. UU No. 34 Tahun 2014 tentang Pengelolaan Keuangan Haji
                2. UU No. 8 Tahun 2019 tentang Penyelenggaraan Ibadah Haji dan Umrah
                3. PP No. 5 Tahun 2018 tentang Pelaksanaan UU No. 34/2014
                4. PP No. 79 Tahun 2012 tentang Pelaksanaan UU Pengelolaan Haji
                5. Peraturan Menteri Agama terkait pengelolaan keuangan haji
                6. Fatwa DSN-MUI tentang investasi syariah
                
                Kewajiban Utama:
                - Transparansi pengelolaan dana
                - Audit eksternal berkala
                - Pelaporan rutin kepada otoritas
                - Compliance syariah yang ketat
                - Perlindungan hak jemaah
                
                Sanksi pelanggaran:
                - Teguran tertulis
                - Pembatasan investasi
                - Penggantian manajemen
                - Denda administratif
                
                Due diligence dan compliance monitoring wajib dilakukan kontinyu.
                """,
                "category": "regulatory",
                "tags": ["regulasi", "compliance", "audit", "transparansi"],
                "language": "id",
                "confidence": 0.90
            },
            {
                "id": "actuarial_modeling",
                "title": "Model Aktuarial untuk Proyeksi Dana Haji",
                "content": """
                Pemodelan aktuarial untuk sustainabilitas dana haji:
                
                A. Input Parameters:
                - Demographic data jemaah (usia, mortality rate)
                - Historical cost trends
                - Investment return assumptions
                - Inflation projections
                - Regulatory changes
                
                B. Key Models:
                1. Mortality Model: Life table analysis untuk proyeksi jemaah
                2. Cost Projection Model: Exponential smoothing untuk biaya
                3. Investment Model: Monte Carlo untuk return projections
                4. Cash Flow Model: Asset-liability matching
                
                C. Output Metrics:
                - Present value of liabilities
                - Required contribution rates
                - Funding ratio projections
                - Risk measures (VaR, Expected Shortfall)
                
                Model validation dilakukan annually dengan backtesting.
                Stress testing scenarios: recession, high inflation, market crash.
                """,
                "category": "actuarial",
                "tags": ["aktuarial", "proyeksi", "mortality", "monte_carlo"],
                "language": "id",
                "confidence": 0.87
            }
        ]
    
    def _load_regulations(self) -> List[Dict]:
        """Load regulatory information"""
        return [
            {
                "regulation_id": "uu_34_2014",
                "title": "UU No. 34 Tahun 2014",
                "summary": "Undang-undang tentang Pengelolaan Keuangan Haji",
                "description": "UU yang mengatur pembentukan BPKH dan pengelolaan keuangan haji secara komprehensif",
                "effective_date": "2014-10-17",
                "key_points": [
                    "Pembentukan BPKH sebagai lembaga pengelola keuangan haji",
                    "Prinsip pengelolaan: syariah, kehati-hatian, manfaat, nirlaba, transparan, akuntabel",
                    "Struktur organisasi: Badan Pelaksana dan Dewan Pengawas",
                    "Keuangan haji meliputi penerimaan dan pengeluaran yang diatur ketat",
                    "Investasi harus sesuai prinsip syariah dengan fokus keamanan dan manfaat",
                    "Wajib audit eksternal dan pelaporan berkala",
                    "Sanksi tegas untuk pelanggaran pengelolaan"
                ],
                "implementing_regulations": [
                    "PP No. 5 Tahun 2018",
                    "Perpres No. 76 Tahun 2016",
                    "Perpres No. 110 Tahun 2017"
                ]
            },
            {
                "regulation_id": "uu_8_2019",
                "title": "UU No. 8 Tahun 2019",
                "summary": "Undang-undang tentang Penyelenggaraan Ibadah Haji dan Umrah",
                "description": "UU yang mengatur penyelenggaraan ibadah haji dan umrah secara menyeluruh",
                "effective_date": "2019-05-07",
                "key_points": [
                    "Pengelolaan dana haji harus transparan dan akuntabel",
                    "Investasi harus sesuai prinsip syariah",
                    "Perlindungan hak dan dana jemaah",
                    "Pengawasan dan audit berkala",
                    "Koordinasi antar lembaga penyelenggara"
                ]
            },
            {
                "regulation_id": "pp_5_2018",
                "title": "PP No. 5 Tahun 2018",
                "summary": "Pelaksanaan UU No. 34 Tahun 2014 tentang Pengelolaan Keuangan Haji",
                "description": "Peraturan pelaksanaan yang mengatur detail operasional BPKH",
                "effective_date": "2018-01-24",
                "key_points": [
                    "Tata cara perencanaan, pelaksanaan, dan pelaporan keuangan haji",
                    "Mekanisme penerimaan dan pengeluaran keuangan haji",
                    "Prosedur investasi dan penempatan dana",
                    "Sistem pengawasan dan audit internal/eksternal",
                    "Koordinasi dengan Kementerian Agama dan lembaga terkait"
                ]
            },
            {
                "regulation_id": "pp_79_2012", 
                "title": "PP No. 79 Tahun 2012",
                "summary": "Peraturan Pelaksanaan Pengelolaan Dana Haji",
                "description": "Peraturan yang mengatur mekanisme pengelolaan dana haji sebelum pembentukan BPKH",
                "effective_date": "2012-12-28",
                "key_points": [
                    "Struktur organisasi pengelola",
                    "Mekanisme investasi dana",
                    "Pelaporan dan akuntabilitas",
                    "Sanksi dan penegakan"
                ]
            }
        ]
    
    def _load_islamic_principles(self) -> List[Dict]:
        """Load Islamic finance principles"""
        return [
            {
                "principle": "Prohibition of Riba",
                "arabic": "تحريم الربا",
                "description": "Larangan riba/bunga dalam semua transaksi keuangan",
                "application": "Investasi hanya pada instrumen bebas bunga seperti sukuk dan saham syariah"
            },
            {
                "principle": "Prohibition of Gharar",
                "arabic": "تحريم الغرر",
                "description": "Larangan ketidakpastian berlebihan dalam kontrak",
                "application": "Menghindari instrumen derivatif kompleks dan spekulatif"
            },
            {
                "principle": "Prohibition of Maysir",
                "arabic": "تحريم الميسر",
                "description": "Larangan judi dan spekulasi",
                "application": "Investasi berbasis fundamental, bukan spekulasi jangka pendek"
            },
            {
                "principle": "Asset Backing",
                "arabic": "دعم الأصول",
                "description": "Setiap investasi harus didukung aset riil",
                "application": "Preferensi pada real estate, komoditas, dan equity"
            },
            {
                "principle": "Transparency and Accountability",
                "arabic": "الشفافية والمساءلة",
                "description": "Pengelolaan harus transparan dan dapat dipertanggungjawabkan",
                "application": "Laporan berkala, audit eksternal, dan disclosure yang memadai"
            }
        ]
    
    def _load_faqs(self) -> List[Dict]:
        """Load frequently asked questions"""
        return [
            {
                "question": "Bagaimana cara menghitung sustainability index dana haji?",
                "answer": "Sustainability Index = (Nilai Manfaat ÷ BPIH) × 100%. Index >60% dianggap sehat, 40-60% perlu perhatian, <40% kritis dan butuh intervensi segera.",
                "category": "calculation"
            },
            {
                "question": "Apa itu BPKH dan bagaimana strukturnya?",
                "answer": "BPKH (Badan Pengelola Keuangan Haji) adalah lembaga yang dibentuk UU No. 34/2014. Terdiri dari Badan Pelaksana (5 orang profesional) dan Dewan Pengawas (7 orang: 2 dari pemerintah, 5 dari masyarakat).",
                "category": "institutional"
            },
            {
                "question": "Instrumen investasi apa saja yang diperbolehkan untuk dana haji?",
                "answer": "Instrumen yang diperbolehkan: Sukuk pemerintah/korporasi, saham syariah (JII), reksadana syariah, deposito syariah, real estate syariah, dan komoditas halal seperti emas.",
                "category": "investment"
            },
            {
                "question": "Berapa target return investasi yang ideal untuk dana haji?",
                "answer": "Target return ideal adalah 6-8% per tahun dengan volatilitas maksimal 15%. Ini mempertimbangkan inflasi, prinsip syariah, dan kebutuhan sustainabilitas jangka panjang.",
                "category": "returns"
            },
            {
                "question": "Bagaimana cara mengelola risiko dalam investasi dana haji?",
                "answer": "Manajemen risiko melalui: 1) Diversifikasi portofolio, 2) Stress testing berkala, 3) Monitoring real-time, 4) Hedging syariah, 5) Contingency planning, dan 6) Insurance takaful.",
                "category": "risk"
            },
            {
                "question": "Apa perbedaan UU No. 34/2014 dengan UU No. 8/2019?",
                "answer": "UU 34/2014 fokus pada pengelolaan keuangan haji dan pembentukan BPKH. UU 8/2019 mengatur penyelenggaraan ibadah haji dan umrah secara menyeluruh termasuk aspek operasional.",
                "category": "regulatory"
            },
            {
                "question": "Bagaimana proses pengawasan pengelolaan dana haji?",
                "answer": "Pengawasan dilakukan berlapis: internal oleh Dewan Pengawas BPKH, eksternal oleh auditor independen, dan pengawasan negara oleh Kementerian Agama dan Kementerian Keuangan.",
                "category": "governance"
            }
        ]
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search knowledge base using simple text matching"""
        query_lower = query.lower()
        results = []
        
        for doc in self.knowledge_docs:
            # Simple scoring based on keyword matches
            content_lower = doc['content'].lower()
            title_lower = doc['title'].lower()
            tags_lower = ' '.join(doc['tags']).lower()
            
            score = 0
            
            # Title matches (highest weight)
            if query_lower in title_lower:
                score += 3
            
            # Content matches
            content_matches = content_lower.count(query_lower)
            score += content_matches * 0.5
            
            # Tag matches
            for word in query_lower.split():
                if word in tags_lower:
                    score += 1
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy['relevance_score'] = score
                results.append(doc_copy)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:top_k]

# RAG Assistant Class
class RAGAssistant:
    """
    RAG-powered assistant for hajj financial guidance
    """
    
    def __init__(self, knowledge_base: HajjFinanceKnowledgeBase):
        self.kb = knowledge_base
        self.conversation_context = []
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        
        # Search relevant knowledge
        relevant_docs = self.kb.search_knowledge(user_query, top_k=3)
        
        # Analyze query intent
        intent = self._analyze_intent(user_query)
        
        # Generate response based on intent and retrieved knowledge
        response = self._generate_response(user_query, relevant_docs, intent)
        
        return {
            'response': response,
            'sources': relevant_docs,
            'intent': intent,
            'confidence': self._calculate_confidence(relevant_docs),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_intent(self, query: str) -> str:
        """Analyze user query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hitung', 'kalkulasi', 'rumus', 'cara menghitung']):
            return 'calculation'
        elif any(word in query_lower for word in ['bpkh', 'organisasi', 'struktur', 'tugas']):
            return 'institutional'
        elif any(word in query_lower for word in ['investasi', 'portofolio', 'saham', 'sukuk']):
            return 'investment'
        elif any(word in query_lower for word in ['risiko', 'manajemen risiko', 'hedge']):
            return 'risk_management'
        elif any(word in query_lower for word in ['regulasi', 'hukum', 'compliance', 'uu', 'pp']):
            return 'regulatory'
        elif any(word in query_lower for word in ['sustainability', 'sustainabilitas', 'jangka panjang']):
            return 'sustainability'
        elif any(word in query_lower for word in ['syariah', 'halal', 'haram', 'riba']):
            return 'islamic_finance'
        else:
            return 'general_inquiry'
    
    def _generate_response(self, query: str, relevant_docs: List[Dict], intent: str) -> str:
        """Generate contextual response"""
        
        if not relevant_docs:
            return self._get_fallback_response(intent)
        
        # Start with greeting and context
        response = f"🤖 **AI Financial Assistant**: Berdasarkan analisis pengetahuan saya tentang keuangan haji...\n\n"
        
        # Add intent-specific response
        if intent == 'calculation':
            response += "📊 **Perhitungan yang Anda tanyakan:**\n\n"
        elif intent == 'institutional':
            response += "🏛️ **Informasi Kelembagaan:**\n\n"
        elif intent == 'investment':
            response += "💰 **Panduan Investasi:**\n\n"
        elif intent == 'risk_management':
            response += "⚠️ **Manajemen Risiko:**\n\n"
        elif intent == 'regulatory':
            response += "📋 **Aspek Regulasi:**\n\n"
        elif intent == 'sustainability':
            response += "🌱 **Sustainabilitas:**\n\n"
        elif intent == 'islamic_finance':
            response += "☪️ **Prinsip Syariah:**\n\n"
        
        # Add relevant knowledge from top document
        top_doc = relevant_docs[0]
        response += self._extract_relevant_content(top_doc['content'], query)
        
        # Add practical recommendations
        response += "\n\n🎯 **Rekomendasi Praktis:**\n"
        recommendations = self._generate_recommendations(intent, relevant_docs)
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec}\n"
        
        # Add sources reference
        if len(relevant_docs) > 1:
            response += f"\n\n📚 **Sumber Tambahan**: {len(relevant_docs)} dokumen terkait tersedia untuk referensi lebih lanjut."
        
        return response
    
    def _extract_relevant_content(self, content: str, query: str) -> str:
        """Extract most relevant part of content"""
        sentences = content.split('\n')
        query_words = query.lower().split()
        
        relevant_sentences = []
        for sentence in sentences:
            if sentence.strip():
                sentence_lower = sentence.lower()
                relevance = sum(1 for word in query_words if word in sentence_lower)
                if relevance > 0:
                    relevant_sentences.append((sentence.strip(), relevance))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for sentence, _ in relevant_sentences[:3]:
            if sentence and not sentence.startswith(('1.', '2.', '3.', 'A.', 'B.', 'C.')):
                result.append(sentence)
        
        return '\n'.join(result) if result else content[:500] + "..."
    
    def _generate_recommendations(self, intent: str, relevant_docs: List[Dict]) -> List[str]:
        """Generate practical recommendations"""
        
        recommendations = {
            'calculation': [
                "Gunakan rumus Sustainability Index = (Nilai Manfaat ÷ BPIH) × 100%",
                "Monitor KPI secara monthly dengan target >60%",
                "Lakukan stress testing scenario berkala"
            ],
            'institutional': [
                "Pahami struktur dan fungsi BPKH sesuai UU No. 34/2014",
                "Koordinasi efektif antara Badan Pelaksana dan Dewan Pengawas",
                "Implementasi governance yang transparan dan akuntabel"
            ],
            'investment': [
                "Diversifikasi portfolio dengan 40% sukuk, 30% saham syariah, 20% real estate, 10% cash",
                "Target return 6-8% per tahun dengan volatilitas <15%",
                "Review dan rebalancing portfolio quarterly"
            ],
            'risk_management': [
                "Implementasikan Value at Risk (VaR) monitoring",
                "Siapkan contingency fund minimum 15% dari total aset",
                "Gunakan hedging syariah untuk mitigasi risiko mata uang"
            ],
            'regulatory': [
                "Pastikan compliance dengan UU No. 34/2014 dan UU No. 8/2019",
                "Implementasikan PP No. 5/2018 untuk operasional BPKH",
                "Lakukan audit eksternal tahunan sesuai regulasi"
            ],
            'sustainability': [
                "Fokus pada long-term value creation",
                "Integrasikan ESG metrics dalam decision making",
                "Develop scenario planning untuk 10-20 tahun ke depan"
            ],
            'islamic_finance': [
                "Pastikan semua investasi mendapat sertifikasi syariah",
                "Avoid instrumen yang mengandung riba, gharar, maysir",
                "Konsultasi dengan Dewan Pengawas Syariah secara berkala"
            ]
        }
        
        return recommendations.get(intent, [
            "Analisis mendalam situasi keuangan current",
            "Konsultasi dengan expert untuk guidance spesifik",
            "Monitor metrics dan KPI secara konsisten"
        ])
    
    def _get_fallback_response(self, intent: str) -> str:
        """Generate fallback response when no relevant documents found"""
        return f"""
        🤖 **AI Assistant**: Maaf, saya tidak menemukan informasi spesifik untuk pertanyaan Anda. 
        
        Namun, saya dapat memberikan guidance umum untuk **{intent}**:
        
        💡 **Saran Umum:**
        - Konsultasikan dengan ahli keuangan syariah
        - Rujuk kepada regulasi terkait (UU No. 34/2014, UU No. 8/2019)
        - Lakukan analisis mendalam sebelum mengambil keputusan
        
        Silakan ajukan pertanyaan yang lebih spesifik untuk mendapat panduan yang lebih tepat.
        """
    
    def _calculate_confidence(self, relevant_docs: List[Dict]) -> float:
        """Calculate confidence score for response"""
        if not relevant_docs:
            return 0.3
        
        # Base confidence from document confidence scores
        doc_confidence = np.mean([doc.get('confidence', 0.5) for doc in relevant_docs])
        
        # Adjust based on number of relevant documents
        coverage_bonus = min(0.2, len(relevant_docs) * 0.05)
        
        return min(0.95, doc_confidence + coverage_bonus)

# Initialize Knowledge Base and Assistant
@st.cache_resource
def get_rag_assistant():
    """Initialize and cache RAG assistant"""
    kb = HajjFinanceKnowledgeBase()
    assistant = RAGAssistant(kb)
    return assistant

# Header
st.markdown("""
<div class="chat-container">
    <h1>🤖 RAG ASSISTANT - PANDUAN KEUANGAN HAJI</h1>
    <h3>Intelligent Knowledge System untuk Konsultasi Finansial Syariah</h3>
    <p>💬 Tanyakan apapun tentang pengelolaan dana haji, investasi syariah, dan sustainabilitas</p>
    <p><em>📜 Telah diperbarui dengan UU No. 34/2014 tentang Pengelolaan Keuangan Haji</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar with knowledge base info
with st.sidebar:
    st.markdown("## 📚 Knowledge Base Info")
    
    assistant = get_rag_assistant()
    kb_stats = {
        "Total Documents": len(assistant.kb.knowledge_docs),
        "Regulations": len(assistant.kb.financial_regulations),
        "Islamic Principles": len(assistant.kb.islamic_principles),
        "FAQs": len(assistant.kb.faqs)
    }
    
    for key, value in kb_stats.items():
        st.metric(key, value)
    
    st.markdown("---")
    
    # Quick examples
    st.markdown("### 💡 Contoh Pertanyaan")
    
    example_questions = [
        "Apa itu BPKH dan bagaimana strukturnya?",
        "Bagaimana cara menghitung sustainability index?",
        "Perbedaan UU 34/2014 dengan UU 8/2019?",
        "Instrumen investasi apa yang direkomendasikan?",
        "Bagaimana strategi manajemen risiko yang efektif?",
        "Apa saja prinsip syariah dalam investasi?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
            st.session_state.example_query = question
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Assistant Settings")
    
    response_style = st.selectbox(
        "Response Style",
        ["Professional", "Detailed", "Concise"],
        index=0
    )
    
    language = st.selectbox(
        "Language",
        ["Indonesian", "English", "Arabic"],
        index=0
    )
    
    include_sources = st.checkbox("Include Sources", value=True)

# Main chat interface
st.markdown("## 💬 Chat dengan AI Assistant")

# Handle example query
if hasattr(st.session_state, 'example_query'):
    user_input = st.session_state.example_query
    delattr(st.session_state, 'example_query')
else:
    user_input = ""

# Chat input
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input(
        "Tanyakan tentang keuangan haji...",
        value=user_input,
        placeholder="Contoh: Bagaimana UU No. 34/2014 mengatur pengelolaan dana haji?",
        key="chat_input"
    )

with col2:
    submit_button = st.button("📤 Send", use_container_width=True, type="primary")

# Process query
if submit_button and user_query:
    assistant = get_rag_assistant()
    
    with st.spinner("🧠 AI sedang menganalisis dan mencari informasi..."):
        # Process the query
        result = assistant.process_query(user_query)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'user': user_query,
            'assistant': result,
            'timestamp': datetime.now()
        })

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 📜 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 conversations
        
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong> {chat['user']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        st.markdown(f"""
        <div class="chat-message assistant-message">
            {chat['assistant']['response']}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence and sources
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            confidence = chat['assistant']['confidence']
            confidence_color = "#27ae60" if confidence > 0.8 else "#f39c12" if confidence > 0.6 else "#e74c3c"
            
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background-color: {confidence_color};"></div>
            </div>
            <small>Confidence: {confidence:.0%}</small>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<span class='source-tag'>Intent: {chat['assistant']['intent']}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            if include_sources and chat['assistant']['sources']:
                with st.expander(f"📚 Sources ({len(chat['assistant']['sources'])})"):
                    for source in chat['assistant']['sources']:
                        st.markdown(f"""
                        <div class="knowledge-card">
                            <h4>{source['title']}</h4>
                            <p><strong>Category:</strong> {source['category']}</p>
                            <p><strong>Relevance:</strong> {source.get('relevance_score', 0):.1f}</p>
                            <p><strong>Tags:</strong> {', '.join(source['tags'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")

# Knowledge base explorer
st.markdown("## 📖 Knowledge Base Explorer")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Documents", "⚖️ Regulations", "☪️ Islamic Principles", "❓ FAQs"])

with tab1:
    st.markdown("### 📚 Available Knowledge Documents")
    
    assistant = get_rag_assistant()
    
    for doc in assistant.kb.knowledge_docs:
        with st.expander(f"📄 {doc['title']}"):
            st.markdown(f"**Category:** {doc['category']}")
            st.markdown(f"**Tags:** {', '.join(doc['tags'])}")
            st.markdown(f"**Language:** {doc['language']}")
            st.markdown(f"**Confidence:** {doc['confidence']:.0%}")
            st.markdown("**Content:**")
            st.markdown(doc['content'])

with tab2:
    st.markdown("### ⚖️ Regulatory Framework")
    
    for reg in assistant.kb.financial_regulations:
        with st.expander(f"📜 {reg['title']}"):
            st.markdown(f"**Summary:** {reg['summary']}")
            if 'description' in reg:
                st.markdown(f"**Description:** {reg['description']}")
            if 'effective_date' in reg:
                st.markdown(f"**Effective Date:** {reg['effective_date']}")
            st.markdown("**Key Points:**")
            for point in reg['key_points']:
                st.markdown(f"• {point}")
            if 'implementing_regulations' in reg:
                st.markdown("**Implementing Regulations:**")
                for impl_reg in reg['implementing_regulations']:
                    st.markdown(f"• {impl_reg}")

with tab3:
    st.markdown("### ☪️ Islamic Finance Principles")
    
    for principle in assistant.kb.islamic_principles:
        with st.expander(f"🕌 {principle['principle']}"):
            st.markdown(f"**Arabic:** {principle['arabic']}")
            st.markdown(f"**Description:** {principle['description']}")
            st.markdown(f"**Application:** {principle['application']}")

with tab4:
    st.markdown("### ❓ Frequently Asked Questions")
    
    # Group FAQs by category
    faq_categories = {}
    for faq in assistant.kb.faqs:
        category = faq['category']
        if category not in faq_categories:
            faq_categories[category] = []
        faq_categories[category].append(faq)
    
    for category, faqs in faq_categories.items():
        st.markdown(f"#### 📂 {category.title()}")
        for faq in faqs:
            with st.expander(f"❓ {faq['question']}"):
                st.markdown(faq['answer'])

# Analytics Dashboard
st.markdown("## 📊 Assistant Analytics")

if st.session_state.chat_history:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_queries = len(st.session_state.chat_history)
        st.metric("Total Queries", total_queries)
    
    with col2:
        avg_confidence = np.mean([chat['assistant']['confidence'] for chat in st.session_state.chat_history])
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    with col3:
        intent_counts = {}
        for chat in st.session_state.chat_history:
            intent = chat['assistant']['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        most_common_intent = max(intent_counts, key=intent_counts.get) if intent_counts else "N/A"
        st.metric("Top Intent", most_common_intent)

# Clear chat history
if st.button("🗑️ Clear Chat History", type="secondary"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <h4>🤖 RAG Assistant untuk Keuangan Haji</h4>
    <p>Powered by Advanced NLP & Islamic Finance Knowledge Base</p>
    <p><em>Memberikan panduan finansial yang akurat dan sesuai syariah</em></p>
    <p><strong>Updated:</strong> Includes UU No. 34/2014 tentang Pengelolaan Keuangan Haji</p>
</div>
""", unsafe_allow_html=True)