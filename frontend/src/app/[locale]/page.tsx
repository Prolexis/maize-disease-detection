"use client";

import React, { useState, useEffect, useRef, use } from "react";
import { useTranslations } from "next-intl";
import { useTheme } from "next-themes";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line
} from "recharts";
import { 
  LayoutDashboard, BrainCircuit, LineChart as ChartIcon, LogOut, Sun, Moon, 
  Languages, Send, Mic, Volume2, VolumeX, Trash2, X, Upload, Download, FileSpreadsheet, 
  FileText, Cpu, AlertTriangle, CheckCircle, HelpCircle, History
} from "lucide-react";

export default function DashboardSPA({ params }: { params: any }) {
  const resolvedParams = typeof params?.then === 'function' ? use(params) : params;
  const locale = (resolvedParams as any)?.locale || 'es';
  const t = useTranslations();
  const { theme, setTheme, resolvedTheme } = useTheme();

  // Authentication State
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [usernameInput, setUsernameInput] = useState("");
  const [passwordInput, setPasswordInput] = useState("");
  const [loginError, setLoginError] = useState("");

  // Tab State
  const [currentTab, setCurrentTab] = useState<"overview" | "cv" | "automl" | "predict_tabular" | "history">("overview");

  // --- Tabular Live Prediction State ---
  const [tabularFeatures, setTabularFeatures] = useState<{ [key: string]: number }>({
    temperature: 24.5,
    humidity: 78.0,
    ph_level: 6.5,
    nitrogen_level: 140.0,
    rainfall_mm: 120.0
  });
  const [predictingTabular, setPredictingTabular] = useState(false);
  const [tabularPredictionResult, setTabularPredictionResult] = useState<any>(null);

  // API Base URL
  const apiBase = "http://localhost:8000/api/v1";

  const fetchModelsMetadata = async (authToken: string) => {
    try {
      const res = await fetch(`${apiBase}/model/metadata`, {
        headers: { "Authorization": `Bearer ${authToken}` }
      });
      if (res.ok) {
        const data = await res.json();
        setModelsMetadata(data);
        if (data.tabular_model) {
          setBestModelInfo({
            name: data.tabular_model.nombre_modelo,
            metrics: {
              accuracy: data.tabular_model.accuracy,
              f1: data.tabular_model["f1-score"] ?? data.tabular_model.f1_score
            }
          });
        }
      }
    } catch (e) {
      console.error("Error fetching models metadata:", e);
    }
  };

  const handlePredictTabular = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredictingTabular(true);
    try {
      const res = await fetch(`${apiBase}/model/predict-tabular`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ features: tabularFeatures, lang: locale })
      });
      if (res.ok) {
        const data = await res.json();
        setTabularPredictionResult(data);
      }
    } catch (err) {
      console.error("Error predicting tabular:", err);
    } finally {
      setPredictingTabular(false);
    }
  };
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [predictingImage, setPredictingImage] = useState(false);
  const [imageResults, setImageResults] = useState<any>(null);
  const [cExportMessage, setCExportMessage] = useState("");

  // --- AutoML Tab States ---
  const [uploadedCsv, setUploadedCsv] = useState<File | null>(null);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState("");
  const [cvFolds, setCvFolds] = useState(5);
  const [splitRatio, setSplitRatio] = useState(80);
  const [seed, setSeed] = useState(42);
  const [alpha, setAlpha] = useState(0.05);
  const [tuningMethod, setTuningMethod] = useState("grid");
  const [activeAutomlTab, setActiveAutomlTab] = useState<"eda" | "models" | "cv" | "tuning" | "stats" | "history">("eda");
  
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  
  // Results
  const [edaResults, setEdaResults] = useState<any>(null);
  const [modelResults, setModelResults] = useState<any>(null);
  const [cvResults, setCvResults] = useState<any>(null);
  const [tuningResults, setTuningResults] = useState<any>(null);
  const [statsResults, setStatsResults] = useState<any>(null);
  const [bootstrapCiData, setBootstrapCiData] = useState<any>(null);
  const [experimentHistory, setExperimentHistory] = useState<any[]>([]);
  const [bestModelInfo, setBestModelInfo] = useState<any>(null);
  const [modelsMetadata, setModelsMetadata] = useState<any>(null);

  // --- Chatbot States ---
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [isRecording, setIsRecording] = useState(false);

  const chatBottomRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);



  // Check login on load
  useEffect(() => {
    setMounted(true);
    const savedToken = localStorage.getItem("token");
    if (savedToken) {
      setToken(savedToken);
      fetchModelsMetadata(savedToken);
    }
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, chatOpen]);

  // Login handler
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError("");
    try {
      const res = await fetch(`${apiBase}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: usernameInput, password: passwordInput })
      });
      if (!res.ok) {
        throw new Error(t("Login.error"));
      }
      const data = await res.json();
      localStorage.setItem("token", data.access_token);
      setToken(data.access_token);
      fetchModelsMetadata(data.access_token);
    } catch (err: any) {
      setLoginError(err.message || t("Login.error"));
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUsernameInput("");
    setPasswordInput("");
  };

  // Fetch Dataset info
  const fetchDefaultDatasetInfo = async (authToken: string) => {
    try {
      const res = await fetch(`${apiBase}/dataset/info`, {
        headers: { "Authorization": `Bearer ${authToken}` }
      });
      if (res.ok) {
        const data = await res.json();
        setCsvColumns(data.columns);
        setTargetCol(data.columns[data.columns.length - 1]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Switch Language — navega hacia la URL con prefijo de locale y escribe la cookie NEXT_LOCALE
  const changeLanguage = (lang: string) => {
    document.cookie = `NEXT_LOCALE=${lang}; path=/; max-age=31536000; SameSite=Lax`;
    const currentPath = window.location.pathname;
    // Reemplaza /es, /en, /pt al inicio de la ruta
    const newPath = currentPath.replace(/^\/(es|en|pt)(\/|$)/, `/${lang}$2`);
    if (newPath !== currentPath) {
      window.location.href = newPath;
    } else {
      window.location.href = `/${lang}`;
    }
  };

  // Quitar imagen seleccionada
  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setImageResults(null);
  };

  // --- Speech Recognition ---
  const toggleRecording = () => {
    if (isRecording) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Su navegador no soporta el reconocimiento de voz.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = locale === "es" ? "es-ES" : locale === "pt" ? "pt-BR" : "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsRecording(true);
    };

    recognition.onresult = (event: any) => {
      const speechToText = event.results[0][0].transcript;
      setChatInput(speechToText);
    };

    recognition.onerror = (event: any) => {
      console.error(event.error);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  // --- Speech Synthesis ---
  const speakText = (text: string) => {
    if (!speechEnabled || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*#`]/g, "");
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = locale === "es" ? "es-ES" : locale === "pt" ? "pt-BR" : "en-US";
    window.speechSynthesis.speak(utterance);
  };

  // --- Chatbot Submission ---
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = chatInput;
    setChatHistory(prev => [...prev, { role: "user", text: userMsg }]);
    setChatInput("");

    try {
      const res = await fetch(`${apiBase}/chat/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ message: userMsg, lang: locale })
      });
      if (res.ok) {
        const data = await res.json();
        setChatHistory(prev => [...prev, { role: "assistant", text: data.response }]);
        speakText(data.response);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // --- Image Diagnosis handler ---
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setImageResults(null);
    }
  };

  const handleImageDiagnose = async () => {
    if (!selectedImage) return;
    setPredictingImage(true);
    setCExportMessage("");

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const res = await fetch(`${apiBase}/model/predict?lang=${locale}`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setImageResults(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setPredictingImage(false);
    }
  };

  // --- TinyML export handler ---
  const handleTinyMLExport = async () => {
    try {
      const res = await fetch(`${apiBase}/api/model/export-c`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setCExportMessage(data.message);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // --- Image report download handler ---
  const handleDownloadImageReport = async (reportType: string) => {
    try {
      const response = await fetch(`${apiBase}/model/report/${reportType}`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` }
      });

      if (!response.ok) throw new Error('Error downloading report');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `reporte_diagnostico.${reportType}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading image report:', error);
    }
  };

  // --- AutoML CSV Upload ---
  const handleCsvChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedCsv(file);
      
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const res = await fetch(`${apiBase}/dataset/upload`, {
          method: "POST",
          headers: { "Authorization": `Bearer ${token}` },
          body: formData
        });
        if (res.ok) {
          const data = await res.json();
          setCsvColumns(data.columns);
          setTargetCol(data.columns[data.columns.length - 1]);
        }
      } catch (e) {
        console.error(e);
      }
    }
  };

  // --- AutoML Run Pipeline (WebSocket integration) ---
  const handleRunAutoML = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus("Iniciando...");
    
    // Generar un ID de cliente aleatorio
    const clientId = Math.random().toString(36).substring(7);
    
    // Conectar WebSocket de progreso
    const ws = new WebSocket(`ws://localhost:8000/api/training/ws/${clientId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "progress") {
        setTrainingProgress(data.percent);
        setTrainingStatus(data.message);
      } else if (data.type === "completed") {
        setTrainingProgress(100);
        setTrainingStatus(data.message);
        setIsTraining(false);
        ws.close();
        // Cargar los resultados de entrenamiento y descriptivos
        fetchAutoMLResults();
      } else if (data.type === "error") {
        setTrainingStatus(data.message);
        setIsTraining(false);
        ws.close();
      }
    };
    
    // Lanzar el trigger HTTP para iniciar entrenamiento en segundo plano
    try {
      await fetch(`${apiBase}/training/train?client_id=${clientId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          split_ratio: splitRatio / 100.0,
          seed: seed,
          cv_folds: cvFolds,
          alpha: alpha,
          tuning_method: tuningMethod
        })
      });
    } catch (e) {
      console.error(e);
      setIsTraining(false);
    }
  };

  // --- Report download handler
  const handleDownloadReport = async (reportType: string) => {
    try {
      const response = await fetch(`${apiBase}/reports/download/${reportType}`, {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });

      if (!response.ok) throw new Error('Error downloading report');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `reporte_fitosanitario_automl.${reportType}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  // Fetch results after pipeline success
  const fetchAutoMLResults = async () => {
    // Cada fetch es independiente: un error no bloquea los demás
    try {
      const resEda = await fetch(`${apiBase}/eda/analyze?target_col=${encodeURIComponent(targetCol)}&lang=${locale}`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (resEda.ok) {
        const edaData = await resEda.json();
        setEdaResults(edaData);
      }
    } catch (e) {
      console.warn("EDA fetch failed:", e);
    }

    try {
      const resLatest = await fetch(`${apiBase}/training/latest?lang=${locale}`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (resLatest.ok) {
        const latestData = await resLatest.json();
        setModelResults(latestData);
        if (latestData.cv_results) setCvResults(latestData.cv_results);
        if (latestData.stats) {
          setStatsResults(latestData.stats);
          setBootstrapCiData(latestData.stats.bootstrap_ci);
        }
        // Guardar info del mejor modelo para el overview
        if (latestData.best_model) {
          setBestModelInfo({
            name: latestData.best_model,
            metrics: latestData.best_model_metrics ?? null
          });
        }
      }
    } catch (e) {
      console.warn("Training latest fetch failed:", e);
    }

    try {
      const resHistory = await fetch(`${apiBase}/training/history`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (resHistory.ok) {
        const histData = await resHistory.json();
        setExperimentHistory(histData.experiments ?? []);
        // También buscar el mejor modelo en el historial si no está seteado
        if (!bestModelInfo && histData.experiments && histData.experiments.length > 0) {
          const latest = histData.experiments[histData.experiments.length - 1];
          setBestModelInfo({ name: latest.best_model_name, metrics: { accuracy: latest.accuracy, f1: latest.f1_score } });
        }
      }
    } catch (e) {
      console.warn("History fetch failed:", e);
    }

    try {
      const resStats = await fetch(`${apiBase}/stats/results?lang=${locale}`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (resStats.ok) {
        const statsData = await resStats.json();
        setStatsResults(statsData);
      }
    } catch (e) {
      console.warn("Stats fetch failed:", e);
    }
  };

  // --- Render Login View ---
  if (!token) {
    return (
      <div 
        className="min-h-screen flex items-center justify-center p-6 font-sans transition-colors duration-200"
        style={{
          background: `linear-gradient(to bottom right, var(--login-grad-start), var(--login-grad-via), var(--login-grad-end))`
        }}
      >
        <div 
          className="w-full max-w-4xl backdrop-blur-md rounded-3xl overflow-hidden shadow-2xl flex flex-col md:flex-row min-h-[500px] border transition-colors duration-200"
          style={{
            backgroundColor: `var(--card-bg)`,
            borderColor: `var(--card-border)`
          }}
        >
          
          {/* Left panel (Branding) */}
          <div className="flex-1 bg-gradient-to-br from-emerald-100/90 to-emerald-200/80 dark:from-emerald-900/70 dark:to-emerald-950/90 p-8 flex flex-col justify-between items-center text-center">
            <div className="w-12 h-12 rounded-full border border-emerald-300/50 dark:border-emerald-400/30 flex items-center justify-center bg-emerald-50/60 dark:bg-emerald-950/50">
              <span className="text-xl">🌽</span>
            </div>
            <div>
              <h1 className="text-3xl font-extrabold tracking-tight uppercase mb-4 leading-none text-slate-800 dark:text-white">
                Detector<br />de <span className="text-emerald-600 dark:text-emerald-400">Enfermedades</span><br />en Hojas de Maíz
              </h1>
              <p className="text-sm text-emerald-800/80 dark:text-emerald-200/80 max-w-sm mx-auto leading-relaxed">
                Inteligencia Artificial para el monitoreo fitosanitario y optimización AutoML.
              </p>
            </div>
            
            {/* Viewfinder simulation */}
            <div className="relative w-28 h-28 flex items-center justify-center border border-emerald-300/40 dark:border-emerald-400/20 rounded-2xl">
              <div className="absolute inset-2 border-2 border-emerald-500/60 dark:border-emerald-400 rounded-lg opacity-40"></div>
              <span className="text-3xl animate-pulse">📷</span>
            </div>
          </div>
          
          {/* Right panel (Form) */}
          <div 
            className="flex-[1.2] p-10 flex flex-col justify-center transition-colors duration-200"
            style={{
              backgroundColor: `var(--card-bg)`
            }}
          >
            {/* Language & Theme selector at top right of right panel */}
            <div className="flex justify-end mb-8 gap-3">
              {/* Theme Toggle */}
              <button 
                onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
                className="p-2 bg-slate-800/80 hover:bg-slate-800 text-slate-300 rounded-lg hover:text-white transition-colors animate-fade-in"
              >
                {!mounted ? <Moon size={18} /> : resolvedTheme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
              </button>
              
              {/* Language Selector */}
              <div className="flex items-center gap-1 bg-slate-850/50 p-1.5 rounded-lg border border-slate-800">
                <Languages size={15} className="text-slate-400 ml-1" />
                <div className="flex gap-1 text-[11px] font-bold">
                  <button onClick={() => changeLanguage("es")} className={`px-2 py-1 rounded ${locale === "es" ? "bg-emerald-600 text-white" : "text-slate-400 hover:text-white"}`}>ES</button>
                  <button onClick={() => changeLanguage("en")} className={`px-2 py-1 rounded ${locale === "en" ? "bg-emerald-600 text-white" : "text-slate-400 hover:text-white"}`}>EN</button>
                  <button onClick={() => changeLanguage("pt")} className={`px-2 py-1 rounded ${locale === "pt" ? "bg-emerald-600 text-white" : "text-slate-400 hover:text-white"}`}>PT</button>
                </div>
              </div>
            </div>

            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-slate-800 dark:text-white">{t("Login.title")}</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{t("Login.subtitle")}</p>
            </div>
            
            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block mb-1">{t("Login.username")}</label>
                <input 
                  type="text" 
                  value={usernameInput}
                  onChange={(e) => setUsernameInput(e.target.value)}
                  placeholder="admin" 
                  className="w-full bg-slate-100/80 dark:bg-slate-800/80 border border-slate-300 dark:border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500 transition-colors"
                  required
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 block mb-1">{t("Login.password")}</label>
                <input 
                  type="password" 
                  value={passwordInput}
                  onChange={(e) => setPasswordInput(e.target.value)}
                  placeholder="••••••••" 
                  className="w-full bg-slate-100/80 dark:bg-slate-800/80 border border-slate-300 dark:border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500 transition-colors"
                  required
                />
              </div>
              
              {loginError && <p className="text-xs text-red-500">{loginError}</p>}
              
              <button 
                type="submit" 
                className="w-full bg-emerald-500 hover:bg-emerald-600 active:scale-[0.98] transition-transform text-white font-bold py-3 px-4 rounded-xl text-sm mt-4 uppercase shadow-lg shadow-emerald-500/20"
              >
                {t("Login.submit")}
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  // --- Main System view (Dashboard Panel) ---
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex font-sans transition-colors duration-200">
      
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col justify-between border-r border-slate-800 shrink-0">
        <div>
          {/* Logo */}
          <div className="p-6 border-b border-slate-800 flex items-center gap-3">
            <span className="text-2xl">🌽</span>
            <div>
              <span className="font-extrabold tracking-tight block">MAIZIA</span>
              <span className="text-xs text-slate-400 block font-medium">AutoML & Diagnóstico</span>
            </div>
          </div>
          
          {/* Tabs */}
          <nav className="p-4 space-y-1">
            <button 
              onClick={() => setCurrentTab("overview")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${currentTab === "overview" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/10" : "text-slate-400 hover:text-white hover:bg-slate-800/50"}`}
            >
              <LayoutDashboard size={18} />
              Panel de Control
            </button>
            <button 
              onClick={() => setCurrentTab("cv")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${currentTab === "cv" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/10" : "text-slate-400 hover:text-white hover:bg-slate-800/50"}`}
            >
              <BrainCircuit size={18} />
              {t("Dashboard.menu_cv")}
            </button>
            <button 
              onClick={() => setCurrentTab("automl")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${currentTab === "automl" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/10" : "text-slate-400 hover:text-white hover:bg-slate-800/50"}`}
            >
              <ChartIcon size={18} />
              {t("Dashboard.menu_automl")}
            </button>
            <button 
              onClick={() => setCurrentTab("predict_tabular")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${currentTab === "predict_tabular" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/10" : "text-slate-400 hover:text-white hover:bg-slate-800/50"}`}
            >
              <Cpu size={18} />
              Predicción Tabular
            </button>
            <button 
              onClick={() => setCurrentTab("history")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${currentTab === "history" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/10" : "text-slate-400 hover:text-white hover:bg-slate-800/50"}`}
            >
              <History size={18} />
              Historial MLOps
            </button>
          </nav>
        </div>
        
        {/* Sidebar Footer (Controls) */}
        <div className="p-4 border-t border-slate-800 flex justify-center">
          <button 
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white font-bold text-xs rounded-xl transition-colors uppercase tracking-wider shadow-lg shadow-red-600/10"
          >
            <LogOut size={14} />
            {t("Common.logout")}
          </button>
        </div>
      </aside>
      
      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="p-6 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-white dark:bg-slate-900 transition-colors">
          <div>
            <h2 className="text-2xl font-extrabold tracking-tight">
              {currentTab === "overview" && t("Dashboard.welcome")}
              {currentTab === "cv" && t("CV.header")}
              {currentTab === "automl" && t("AutoML.header")}
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {currentTab === "overview" && t("Dashboard.desc")}
              {currentTab === "cv" && "Diagnóstico foliar mediante Redes Convolucionales"}
              {currentTab === "automl" && "Automatización, entrenamiento y validaciones estadísticas"}
            </p>
          </div>

          {/* Theme & Language Controls at the top right of Main Content */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <button 
              onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
              className="p-2.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-850 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800 rounded-xl transition-colors shadow-sm"
              title={resolvedTheme === "dark" ? t("Common.theme_light") : t("Common.theme_dark")}
            >
              {!mounted ? <Moon size={18} /> : resolvedTheme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            
            {/* Language Selector */}
            <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-850 p-1.5 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
              <Languages size={15} className="text-slate-500 dark:text-slate-400 ml-1" />
              <div className="flex gap-1 text-[11px] font-bold">
                <button onClick={() => changeLanguage("es")} className={`px-2.5 py-1 rounded-lg transition-colors ${locale === "es" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"}`}>ES</button>
                <button onClick={() => changeLanguage("en")} className={`px-2.5 py-1 rounded-lg transition-colors ${locale === "en" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"}`}>EN</button>
                <button onClick={() => changeLanguage("pt")} className={`px-2.5 py-1 rounded-lg transition-colors ${locale === "pt" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"}`}>PT</button>
              </div>
            </div>
          </div>
        </header>
        
        {/* Tab Components */}
        <div className="p-6 max-w-6xl w-full mx-auto space-y-6">
          
          {/* TAB 1: OVERVIEW */}
          {currentTab === "overview" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl flex flex-col justify-between">
                  <div>
                    <h3 className="text-lg font-bold flex items-center gap-2 mb-2 text-emerald-600 dark:text-emerald-400">
                      <BrainCircuit />
                      {t("Dashboard.menu_cv")}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed mb-4">
                      {t("Dashboard.card_cv_desc")}
                    </p>
                    <div className="p-3 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800/50 rounded-xl mb-4 text-xs flex items-center justify-between">
                      <span className="font-semibold text-blue-700 dark:text-blue-300 flex items-center gap-1">
                        🏆 Mejor Modelo: {modelsMetadata?.vision_model?.best_model || "EfficientNetB0"}
                      </span>
                      <span className="font-bold text-blue-600 dark:text-blue-400">
                        {((modelsMetadata?.vision_model?.best_accuracy || 0.983) * 100).toFixed(1)}% Accuracy
                      </span>
                    </div>
                  </div>
                  <button 
                    onClick={() => setCurrentTab("cv")}
                    className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-2.5 rounded-xl text-sm transition-transform active:scale-[0.98]"
                  >
                    {t("Dashboard.menu_cv")}
                  </button>
                </div>
                
                <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl flex flex-col justify-between">
                  <div>
                    <h3 className="text-lg font-bold flex items-center gap-2 mb-2 text-emerald-600 dark:text-emerald-400">
                      <ChartIcon />
                      {t("Dashboard.menu_automl")}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed mb-4">
                      {t("Dashboard.card_automl_desc")}
                    </p>
                    <div className="p-3 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/50 rounded-xl mb-4 text-xs flex items-center justify-between">
                      <span className="font-semibold text-emerald-700 dark:text-emerald-300 flex items-center gap-1">
                        🏆 Mejor Modelo: {modelsMetadata?.tabular_model?.nombre_modelo || bestModelInfo?.name || "Random Forest (Clásico)"}
                      </span>
                      <span className="font-bold text-emerald-600 dark:text-emerald-400">
                        {(((modelsMetadata?.tabular_model?.accuracy || bestModelInfo?.metrics?.accuracy || 0.8917)) * 100).toFixed(1)}% Accuracy
                      </span>
                    </div>
                  </div>
                  <button 
                    onClick={() => setCurrentTab("automl")}
                    className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-2.5 rounded-xl text-sm transition-transform active:scale-[0.98]"
                  >
                    {t("Dashboard.menu_automl")}
                  </button>
                </div>
              </div>

              {/* Modelos Activos en el Sistema (Visión + Tabular) */}
              <div className="space-y-4">
                <h3 className="text-md font-bold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                  <span>🤖</span> Modelos Activos en Producción
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Tarjeta Modelo de Visión */}
                  <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/20 border border-blue-200 dark:border-blue-800/40 rounded-2xl">
                    <div className="flex items-center justify-between flex-wrap gap-4 mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center">
                          <span className="text-xl">📸</span>
                        </div>
                        <div>
                          <p className="text-xs font-bold text-blue-700 dark:text-blue-400 uppercase tracking-wider">Mejor Modelo de Visión</p>
                          <p className="text-xl font-extrabold text-slate-800 dark:text-white">
                            {modelsMetadata?.vision_model?.best_model || "EfficientNetB0"}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="block text-2xl font-extrabold text-blue-600 dark:text-blue-400">
                          {((modelsMetadata?.vision_model?.best_accuracy || 0.983) * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-slate-500 font-semibold">Exactitud CV</span>
                      </div>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-300 pt-2 border-t border-blue-200/60 dark:border-blue-800/40 flex items-center justify-between">
                      <span>Estrategia: Ensamble (3 CNNs con Consenso)</span>
                      <span className="bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300 px-2 py-0.5 rounded-md font-semibold">Visión Foliares</span>
                    </div>
                  </div>

                  {/* Tarjeta Modelo Tabular / Pipeline */}
                  <div className="p-6 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/30 dark:to-teal-950/20 border border-emerald-200 dark:border-emerald-800/40 rounded-2xl">
                    <div className="flex items-center justify-between flex-wrap gap-4 mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                          <span className="text-xl">📊</span>
                        </div>
                        <div>
                          <p className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">Mejor Pipeline Tabular (AutoML)</p>
                          <p className="text-xl font-extrabold text-slate-800 dark:text-white">
                            {modelsMetadata?.tabular_model?.nombre_modelo || bestModelInfo?.name || "Random Forest (Clásico)"}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="block text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">
                          {(((modelsMetadata?.tabular_model?.accuracy || bestModelInfo?.metrics?.accuracy || 0.8917)) * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-slate-500 font-semibold">Accuracy</span>
                      </div>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-300 pt-2 border-t border-emerald-200/60 dark:border-emerald-800/40 flex items-center justify-between">
                      <span>F1-Score: {(((modelsMetadata?.tabular_model?.["f1-score"] || bestModelInfo?.metrics?.f1 || 0.8933)) * 100).toFixed(1)}%</span>
                      <span className="bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 px-2 py-0.5 rounded-md font-semibold">AutoML Tabular</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* TAB 2: COMPUTER VISION (DIAGNOSIS) */}
          {currentTab === "cv" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Image Input side */}
              <div className="lg:col-span-1 space-y-6">
                <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl">
                  <h3 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                    {t.has("CV.upload_label_title") ? t("CV.upload_label_title") : (locale === "pt" ? "Carregar Imagem" : locale === "en" ? "Upload Image" : "Cargar Imagen")}
                  </h3>
                  
                  {/* Zona de carga con botón X para quitar imagen */}
                  <div className="relative">
                    <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-emerald-500 dark:hover:border-emerald-500 rounded-xl p-6 cursor-pointer transition-colors relative min-h-[200px]">
                      <input 
                        type="file" 
                        accept="image/*" 
                        onChange={handleImageChange} 
                        className="hidden" 
                      />
                      
                      {imagePreview ? (
                        <img src={imagePreview} alt="Preview" className="absolute inset-0 w-full h-full object-cover rounded-xl" />
                      ) : (
                        <div className="text-center text-slate-400">
                          <Upload size={32} className="mx-auto mb-2 text-slate-400" />
                          <span className="text-xs font-semibold">{t("CV.upload_label")}</span>
                        </div>
                      )}
                    </label>

                    {/* Botón X para quitar imagen */}
                    {imagePreview && (
                      <button
                        onClick={clearImage}
                        type="button"
                        className="absolute top-2 right-2 z-10 w-7 h-7 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-lg transition-transform active:scale-90"
                        title="Quitar imagen"
                      >
                        <X size={14} />
                      </button>
                    )}
                  </div>
                  
                  {selectedImage && (
                    <button 
                      onClick={handleImageDiagnose}
                      disabled={predictingImage}
                      className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-3 rounded-xl text-sm mt-4 uppercase tracking-wider shadow-lg shadow-emerald-500/10"
                    >
                      {predictingImage ? "Analizando..." : t("CV.btn_diagnose")}
                    </button>
                  )}
                </div>
              </div>
              
              {/* Output Results side */}
              <div className="lg:col-span-2 space-y-6">
                {imageResults ? (() => {
                  const topVisionModelEntry = imageResults?.predictions ? Object.entries(imageResults.predictions).reduce((best: any, current: any) => {
                    return (!best || current[1].confidence > best[1].confidence) ? current : best;
                  }, null) : null;

                  return (
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-6">
                      
                      {/* Consensus header */}
                      <div className="p-4 rounded-xl border flex items-center justify-between bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800/30">
                        <div>
                          <h4 className="font-extrabold text-lg flex items-center gap-2">
                            <CheckCircle className="text-emerald-500" />
                            {imageResults.consensus_reached ? `Diagnóstico: ${imageResults.consensus_diagnosis}` : t("CV.no_consensus")}
                          </h4>
                          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                            {imageResults.consensus_reached ? "Consenso unánime alcanzado por el consorcio." : t("CV.no_consensus_desc")}
                          </p>
                          {imageResults.interpretation && (
                            <p className="text-sm text-emerald-800 dark:text-emerald-300 mt-2 font-medium bg-emerald-100/50 dark:bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-200/50 dark:border-emerald-800/30">
                              {imageResults.interpretation}
                            </p>
                          )}
                        </div>
                      </div>

                      {/* Tarjeta Destacada del Mejor Modelo de Visión */}
                      {topVisionModelEntry && (
                        <div className="p-4 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/40 dark:to-teal-950/30 border border-emerald-300 dark:border-emerald-700/50 rounded-xl flex items-center justify-between flex-wrap gap-4 shadow-sm">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                              <span className="text-xl">🏆</span>
                            </div>
                            <div>
                              <p className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
                                Mejor Modelo de Visión (Mayor Confianza)
                              </p>
                              <p className="text-lg font-extrabold text-slate-800 dark:text-white">
                                {topVisionModelEntry[0]}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <span className="block text-xl font-extrabold text-emerald-600 dark:text-emerald-400">
                                {(topVisionModelEntry[1].confidence * 100).toFixed(2)}%
                              </span>
                              <span className="text-xs text-slate-500 font-semibold">Nivel de Confianza</span>
                            </div>
                            <div className="text-right pl-4 border-l border-emerald-200 dark:border-emerald-800">
                              <span className="block text-sm font-bold text-slate-800 dark:text-slate-200">
                                {topVisionModelEntry[1].class}
                              </span>
                              <span className="text-xs text-slate-500 font-semibold">Diagnóstico</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Recommendations */}
                      {imageResults.recommendations && (
                        <div className="space-y-2">
                          <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">{t("CV.recommendations")}</h4>
                          <div className="p-4 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl text-sm leading-relaxed whitespace-pre-line text-slate-700 dark:text-slate-300">
                            {imageResults.recommendations}
                          </div>
                        </div>
                      )}
                      
                      {/* Probability charts */}
                      <div className="space-y-4">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">Probabilidades por Modelo</h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          {Object.entries(imageResults.predictions).map(([name, pred]: any) => {
                            const chartData = [
                              { name: "Mancha gris", value: pred.probabilities[0] },
                              { name: "Roña común", value: pred.probabilities[1] },
                              { name: "Tizón norte", value: pred.probabilities[2] },
                              { name: "Sano", value: pred.probabilities[3] }
                            ];
                            
                            return (
                              <div key={name} className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/30">
                                <h5 className="text-xs font-bold text-slate-400 mb-3 text-center">{name}</h5>
                                <div className="h-32">
                                  <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData}>
                                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                      <XAxis 
                                        dataKey="name" 
                                        tick={{ fontSize: 10 }} 
                                        angle={-45} 
                                        textAnchor="end" 
                                        height={40} 
                                      />
                                      <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
                                      <Tooltip 
                                        formatter={(value: any) => [`${(value * 100).toFixed(2)}%`, 'Probabilidad']} 
                                        contentStyle={{ fontSize: '10px' }}
                                      />
                                      <Bar dataKey="value" fill="#10b981" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                  </ResponsiveContainer>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      
                      {/* Individual predictions list */}
                      <div className="space-y-4">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">{t("CV.model_predictions")}</h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          {Object.entries(imageResults.predictions).map(([name, pred]: any) => {
                            const isBest = topVisionModelEntry && topVisionModelEntry[0] === name;
                            return (
                              <div 
                                key={name} 
                                className={`p-4 border rounded-xl text-center space-y-1 relative transition-all ${
                                  isBest 
                                    ? "border-emerald-500 bg-emerald-50/40 dark:bg-emerald-950/40 shadow-sm ring-1 ring-emerald-500/30" 
                                    : "border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/30"
                                }`}
                              >
                                {isBest && (
                                  <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 bg-emerald-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-sm flex items-center gap-1">
                                    🏆 Mejor Modelo
                                  </span>
                                )}
                                <span className="text-xs font-bold text-slate-400 block pt-1">{name}</span>
                                <span className="text-base font-extrabold text-emerald-600 dark:text-emerald-400 block">{pred.class}</span>
                                <span className="text-xs font-semibold text-slate-500">{`${(pred.confidence * 100).toFixed(2)}%`}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      
                      {/* Summary table */}
                      <div className="space-y-2">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">Resumen de Predicciones</h4>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs border-collapse">
                            <thead>
                              <tr className="bg-slate-50 dark:bg-slate-950">
                                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left font-bold text-slate-600 dark:text-slate-300">Modelo</th>
                                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left font-bold text-slate-600 dark:text-slate-300">Predicción</th>
                                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left font-bold text-slate-600 dark:text-slate-300">Confianza</th>
                                <th className="border border-slate-200 dark:border-slate-700 p-2 text-left font-bold text-slate-600 dark:text-slate-300">Estado</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(imageResults.predictions).map(([name, pred]: any) => {
                                const isBest = topVisionModelEntry && topVisionModelEntry[0] === name;
                                return (
                                  <tr key={name} className={isBest ? "bg-emerald-50/30 dark:bg-emerald-950/20 font-medium" : ""}>
                                    <td className="border border-slate-200 dark:border-slate-700 p-2 font-semibold">
                                      {name} {isBest && <span className="ml-1" title="Mejor Modelo">🏆</span>}
                                    </td>
                                    <td className="border border-slate-200 dark:border-slate-700 p-2">{pred.class}</td>
                                    <td className="border border-slate-200 dark:border-slate-700 p-2">{(pred.confidence * 100).toFixed(2)}%</td>
                                    <td className={`border border-slate-200 dark:border-slate-700 p-2 font-bold ${pred.class === "Sano" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`}>
                                      {pred.class === "Sano" ? "Saludable" : "Infectado"}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      
                      {/* Report download buttons */}
                      <div className="pt-4 border-t border-slate-200 dark:border-slate-800">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">Descargar Reportes</h4>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                          <button 
                            onClick={() => handleDownloadImageReport("pdf")}
                            className="bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs py-2.5 px-4 rounded-xl flex items-center justify-center gap-2"
                          >
                            <FileText size={14} />
                            PDF
                          </button>
                          <button 
                            onClick={() => handleDownloadImageReport("docx")}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs py-2.5 px-4 rounded-xl flex items-center justify-center gap-2"
                          >
                            <FileSpreadsheet size={14} />
                            DOCX
                          </button>
                          <button 
                            onClick={() => handleDownloadImageReport("xlsx")}
                            className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs py-2.5 px-4 rounded-xl flex items-center justify-center gap-2"
                          >
                            <FileSpreadsheet size={14} />
                            XLSX
                          </button>
                        </div>
                      </div>
                      
                    </div>
                  );
                })() : (
                  <div className="h-64 flex flex-col items-center justify-center border border-dashed border-slate-200 dark:border-slate-800 rounded-2xl text-slate-400">
                    <HelpCircle size={32} className="text-slate-400 mb-2" />
                    <span className="text-xs font-semibold">Esperando análisis...</span>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* TAB 3: AUTOML TABULAR */}
          {currentTab === "automl" && (
            <div className="space-y-6">
              
              {/* Configuration Panel */}
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-6">
                
                {/* Upload & info row */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                  <div>
                    <h3 className="font-bold text-sm text-slate-400 uppercase tracking-wider">{t("AutoML.upload_csv")}</h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Cargue datos personalizados o use el dataset por defecto de maiz.</p>
                  </div>
                  <label className="bg-slate-100 hover:bg-slate-200 text-slate-800 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 font-bold text-xs py-2.5 px-4 rounded-xl cursor-pointer flex items-center gap-2 transition-colors shadow-sm">
                    <Upload size={14} />
                    Cargar CSV
                    <input 
                      type="file" 
                      accept=".csv" 
                      onChange={handleCsvChange} 
                      className="hidden" 
                    />
                  </label>
                </div>
                
                {/* Form fields */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-4 border-t border-slate-200 dark:border-slate-800">
                  <div>
                    <label className="text-xs font-bold text-slate-400 block mb-1">{t("AutoML.target_col")}</label>
                    <select 
                      value={targetCol}
                      onChange={(e) => setTargetCol(e.target.value)}
                      className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-emerald-500"
                    >
                      {csvColumns.map((col) => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="text-xs font-bold text-slate-400 block mb-1">{t("AutoML.cv_folds")} ({cvFolds})</label>
                    <input 
                      type="range" 
                      min="3" 
                      max="10" 
                      value={cvFolds}
                      onChange={(e) => setCvFolds(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="text-xs font-bold text-slate-400 block mb-1">{t("AutoML.train_pct")} ({splitRatio}%)</label>
                    <input 
                      type="range" 
                      min="60" 
                      max="90" 
                      value={splitRatio}
                      onChange={(e) => setSplitRatio(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="text-xs font-bold text-slate-400 block mb-1">{t("AutoML.search_method")}</label>
                    <div className="flex gap-2">
                      <label className="flex items-center gap-1.5 text-xs font-semibold cursor-pointer">
                        <input type="radio" checked={tuningMethod === "grid"} onChange={() => setTuningMethod("grid")} />
                        Grid
                      </label>
                      <label className="flex items-center gap-1.5 text-xs font-semibold cursor-pointer">
                        <input type="radio" checked={tuningMethod === "random"} onChange={() => setTuningMethod("random")} />
                        Random
                      </label>
                    </div>
                  </div>
                </div>
                
                <button 
                  onClick={handleRunAutoML}
                  disabled={isTraining}
                  className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-3 rounded-xl text-sm uppercase tracking-wider shadow-lg shadow-emerald-500/10"
                >
                  {isTraining ? "Entrenando..." : t("AutoML.btn_run")}
                </button>
                
                {/* Live Progress feedback */}
                {isTraining && (
                  <div className="space-y-2 pt-4 border-t border-slate-200 dark:border-slate-800">
                    <div className="flex justify-between items-center text-xs font-semibold">
                      <span>{trainingStatus}</span>
                      <span>{trainingProgress}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 dark:bg-slate-950 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 transition-all duration-300" style={{ width: `${trainingProgress}%` }}></div>
                    </div>
                  </div>
                )}
                
              </div>
              
              {/* AutoML Output Dashboard */}
              {edaResults && modelResults && (
                <div className="space-y-6">
                  {/* AutoML Tabs Navigation */}
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {[
                      { key: "eda", label: t("AutoML.tab_eda") },
                      { key: "models", label: t("AutoML.tab_models") },
                      { key: "cv", label: t("AutoML.tab_cv") },
                      { key: "tuning", label: t("AutoML.tab_tuning") },
                      { key: "stats", label: t("AutoML.tab_stats") }
                    ].map((tab) => (
                      <button
                        key={tab.key}
                        onClick={() => setActiveAutomlTab(tab.key as any)}
                        className={`px-4 py-2 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
                          activeAutomlTab === tab.key
                            ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                            : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-emerald-500 hover:text-emerald-600 dark:hover:text-emerald-400"
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>

                  {/* Tab Contents */}
                  <div className="space-y-6">
                    {/* EDA Tab */}
                    {activeAutomlTab === "eda" && (
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                          🔬 {t("AutoML.eda_section")}
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                          <div className="bg-slate-50 dark:bg-slate-950/40 rounded-xl p-3 text-center">
                            <span className="block text-2xl font-extrabold text-emerald-500">
                              {edaResults.num_duplicates ?? 0}
                            </span>
                            <span className="text-xs text-slate-500 mt-0.5 block">Duplicados</span>
                          </div>
                          <div className="bg-slate-50 dark:bg-slate-950/40 rounded-xl p-3 text-center">
                            <span className="block text-2xl font-extrabold text-amber-500">
                              {Object.keys(edaResults.imputed_nulls ?? {}).length}
                            </span>
                            <span className="text-xs text-slate-500 mt-0.5 block">Cols. con Nulos</span>
                          </div>
                          <div className="bg-slate-50 dark:bg-slate-950/40 rounded-xl p-3 text-center">
                            <span className="block text-2xl font-extrabold text-rose-500">
                              {edaResults.multivariate_outliers ?? 0}
                            </span>
                            <span className="text-xs text-slate-500 mt-0.5 block">Outliers Multivariados</span>
                          </div>
                          <div className="bg-slate-50 dark:bg-slate-950/40 rounded-xl p-3 text-center">
                            <span className="block text-2xl font-extrabold text-blue-500">
                              {edaResults.transformed_cols?.length ?? 0}
                            </span>
                            <span className="text-xs text-slate-500 mt-0.5 block">Cols. Transformadas</span>
                          </div>
                        </div>

                        {Object.keys(edaResults.imputed_nulls ?? {}).length > 0 && (
                          <div className="mb-6">
                            <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Imputación de Nulos</h5>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b border-slate-200 dark:border-slate-700">
                                    {["Columna", "Cantidad", "Método"].map(h => (
                                      <th key={h} className="text-left py-2 px-2 text-slate-400 font-semibold">{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(edaResults.imputed_nulls).map(([col, info]: any) => (
                                    <tr key={col} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                                      <td className="py-1.5 px-2 font-medium text-slate-700 dark:text-slate-300">{col}</td>
                                      <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{info.count}</td>
                                      <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{info.method}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {edaResults.descriptive_stats && (
                          <div>
                            <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Estadísticas Descriptivas</h5>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b border-slate-200 dark:border-slate-700">
                                    {["Variable", "Media", "Std", "Mín", "Máx"].map(h => (
                                      <th key={h} className="text-left py-2 px-2 text-slate-400 font-semibold">{h}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(edaResults.descriptive_stats).map(([col, stats]: any) => {
                                    const meanVal = stats?.mean ?? stats?.media ?? stats?.Media;
                                    const stdVal = stats?.std ?? stats?.Std ?? stats?.desviación ?? stats?.["Desv. Estándar"];
                                    const minVal = stats?.min ?? stats?.Min ?? stats?.["mín"] ?? stats?.["Mín"] ?? stats?.p25;
                                    const maxVal = stats?.max ?? stats?.Max ?? stats?.["máx"] ?? stats?.["Máx"] ?? stats?.p75;
                                    return (
                                      <tr key={col} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                                        <td className="py-1.5 px-2 font-medium text-slate-700 dark:text-slate-300">{col}</td>
                                        <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof meanVal === 'number' ? meanVal.toFixed(3) : '—'}</td>
                                        <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof stdVal === 'number' ? stdVal.toFixed(3) : '—'}</td>
                                        <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof minVal === 'number' ? minVal.toFixed(3) : '—'}</td>
                                        <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof maxVal === 'number' ? maxVal.toFixed(3) : '—'}</td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                        {edaResults.interpretation && (
                          <div className="mt-6 p-4 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/30 rounded-xl">
                            <h5 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2">Interpretación de Calidad y EDA</h5>
                            <p className="text-xs text-slate-600 dark:text-slate-350 leading-relaxed whitespace-pre-line">{edaResults.interpretation}</p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Models Tab */}
                    {activeAutomlTab === "models" && (
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                          📊 {t("AutoML.tab_models")}
                        </h4>
                        <div className="space-y-6">
                          {/* Exactitud Modelos Chart */}
                          <div className="h-80 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={modelResults.metrics ? 
                                  Object.entries(modelResults.metrics).map(([name, metrics]: any) => ({
                                    name,
                                    Accuracy: metrics.accuracy,
                                    F1: metrics.f1
                                  })) 
                                  : [
                                    { name: "Logistic Regression", Accuracy: 0.82, F1: 0.80 },
                                    { name: "Random Forest", Accuracy: 0.89, F1: 0.88 },
                                    { name: "MLP Neural Network", Accuracy: 0.91, F1: 0.90 },
                                    { name: "Voting Hybrid", Accuracy: 0.93, F1: 0.92 },
                                    { name: "Stacking Hybrid", Accuracy: 0.95, F1: 0.94 }
                                  ]
                                }
                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis domain={[0, 1]} />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="Accuracy" fill="#10b981" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="F1" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>

                          {modelResults.best_model && (
                            <div className="p-4 border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/20 rounded-xl">
                              <h5 className="font-bold text-sm text-emerald-700 dark:text-emerald-400 mb-3">
                                🏆 Mejor Modelo: {modelResults.best_model}
                              </h5>
                              {modelResults.best_model_metrics && (
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                                    <span className="block text-slate-500 dark:text-slate-400">Accuracy</span>
                                    <span className="block text-lg font-bold text-emerald-600 dark:text-emerald-400">
                                      {(modelResults.best_model_metrics.accuracy * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                                    <span className="block text-slate-500 dark:text-slate-400">Precision</span>
                                    <span className="block text-lg font-bold text-blue-600 dark:text-blue-400">
                                      {(modelResults.best_model_metrics.precision * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                                    <span className="block text-slate-500 dark:text-slate-400">Recall</span>
                                    <span className="block text-lg font-bold text-amber-600 dark:text-amber-400">
                                      {(modelResults.best_model_metrics.recall * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                  <div className="p-3 bg-white dark:bg-slate-800 rounded-lg">
                                    <span className="block text-slate-500 dark:text-slate-400">F1 Score</span>
                                    <span className="block text-lg font-bold text-rose-600 dark:text-rose-400">
                                      {(modelResults.best_model_metrics.f1 * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                          {modelResults.interpretations && (
                            <div className="p-4 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/30 rounded-xl mt-4">
                              <h5 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2">Interpretación del Entrenamiento</h5>
                              <p className="text-xs text-slate-600 dark:text-slate-355 leading-relaxed whitespace-pre-line">{modelResults.interpretations}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Cross Validation Tab */}
                    {activeAutomlTab === "cv" && (
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                          🔄 {t("AutoML.tab_cv")}
                        </h4>
                        {modelResults && modelResults.cv_results ? (
                          <div className="space-y-4">
                            <div className="h-80 w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                  data={Object.entries(modelResults.cv_results).map(([name, scores]: any) => ({
                                    name,
                                    ...scores.reduce((acc: any, score: number, idx: number) => {
                                      acc[`Fold ${idx + 1}`] = score;
                                      return acc;
                                    }, {})
                                  }))}
                                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="name" />
                                  <YAxis domain={[0, 1]} />
                                  <Tooltip />
                                  <Legend />
                                  {Array.from({ length: Math.min(cvFolds, 5) }, (_, i) => (
                                    <Bar key={`fold${i}`} dataKey={`Fold ${i + 1}`} fill={["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"][i % 5]} radius={[4, 4, 0, 0]} />
                                  ))}
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                            {modelResults.cv_interpretation && (
                              <div className="p-4 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/30 rounded-xl mt-4">
                                <h5 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2">Interpretación de Validación Cruzada</h5>
                                <p className="text-xs text-slate-600 dark:text-slate-355 leading-relaxed whitespace-pre-line">{modelResults.cv_interpretation}</p>
                              </div>
                            )}
                          </div>
                        ) : (
                          <p className="text-sm text-slate-500">No hay resultados de validación cruzada disponibles.</p>
                        )}
                      </div>
                    )}

                    {/* Hyperparameter Tuning Tab */}
                    {activeAutomlTab === "tuning" && (
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-6">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                          🔧 {t("AutoML.tab_tuning")}
                        </h4>
                        {modelResults && modelResults.tuning_results ? (
                          <>
                            {/* Summary Cards */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50 dark:bg-slate-950/30">
                                <span className="text-xs font-semibold text-slate-500 block">Método</span>
                                <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400 uppercase">{modelResults.tuning_results.method}</span>
                              </div>
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50 dark:bg-slate-950/30">
                                <span className="text-xs font-semibold text-slate-500 block">Tiempo (s)</span>
                                <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{modelResults.tuning_results.search_time.toFixed(2)}</span>
                              </div>
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50 dark:bg-slate-950/30">
                                <span className="text-xs font-semibold text-slate-500 block">Antes</span>
                                <span className="text-sm font-extrabold text-slate-700 dark:text-slate-300">{(modelResults.tuning_results.accuracy_before * 100).toFixed(2)}%</span>
                              </div>
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50 dark:bg-slate-950/30">
                                <span className="text-xs font-semibold text-slate-500 block">Después</span>
                                <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{(modelResults.tuning_results.accuracy_after * 100).toFixed(2)}%</span>
                              </div>
                            </div>
                            
                            {/* Best Params */}
                            <div>
                              <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Mejores Parámetros</h5>
                              <pre className="text-[10px] whitespace-pre-wrap bg-slate-50 dark:bg-slate-950/30 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
                                {JSON.stringify(modelResults.tuning_results.best_params, null, 2)}
                              </pre>
                            </div>
                            {modelResults.tuning_interpretation && (
                              <div className="p-4 border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/30 rounded-xl mt-4">
                                <h5 className="font-bold text-xs text-slate-400 uppercase tracking-wider mb-2">Interpretación del Ajuste</h5>
                                <p className="text-xs text-slate-600 dark:text-slate-355 leading-relaxed whitespace-pre-line">{modelResults.tuning_interpretation}</p>
                              </div>
                            )}
                          </>
                        ) : (
                          <p className="text-sm text-slate-500">No hay resultados de tuning disponibles.</p>
                        )}
                      </div>
                    )}

                    {/* Statistical Tests Tab */}
                    {activeAutomlTab === "stats" && (
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-6">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider mb-4">
                          📈 {t("AutoML.tab_stats")}
                        </h4>
                        
                        {/* Statistical Test Results */}
                        {statsResults && (
                          <div className="space-y-6">
                            {/* 1. Global Test (ANOVA/Friedman) */}
                            <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/20">
                              <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Test Global</h5>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div>
                                  <span className="text-xs font-semibold text-slate-500 block">Tipo de Test</span>
                                  <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{statsResults.test_type}</span>
                                </div>
                                <div>
                                  <span className="text-xs font-semibold text-slate-500 block">Estadístico</span>
                                  <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{statsResults.overall_stat?.toFixed(4) ?? "—"}</span>
                                </div>
                                <div>
                                  <span className="text-xs font-semibold text-slate-500 block">Valor p</span>
                                  <span className={`text-sm font-extrabold ${statsResults.overall_pval < 0.05 ? "text-rose-600 dark:text-rose-400" : "text-slate-600 dark:text-slate-300"}`}>
                                    {statsResults.overall_pval?.toFixed(4) ?? "—"}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-xs font-semibold text-slate-500 block">Paramétrico</span>
                                  <span className={`text-sm font-extrabold ${statsResults.use_parametric ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>
                                    {statsResults.use_parametric ? "Sí" : "No"}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* 2. Assumptions (Shapiro-Wilk + Levene) */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/20">
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Normalidad (Shapiro-Wilk)</h5>
                                <div className="space-y-2 max-h-40 overflow-y-auto">
                                  {Object.entries(statsResults.shapiro_pvals ?? {}).map(([model, pval]: [string, any]) => (
                                    <div key={model} className="flex justify-between items-center">
                                      <span className="text-xs font-medium text-slate-700 dark:text-slate-300">{model}</span>
                                      <span className={`text-xs font-bold ${pval < 0.05 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                                        p = {pval.toFixed(4)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/20">
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Homocedasticidad (Levene)</h5>
                                <div className="flex justify-between items-center">
                                  <span className="text-xs font-medium text-slate-700 dark:text-slate-300">Valor p</span>
                                  <span className={`text-xs font-bold ${statsResults.levene_pval < 0.05 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                                    {statsResults.levene_pval?.toFixed(4) ?? "—"}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* 3. Post-Hoc Results */}
                            {statsResults.posthoc_results && Object.keys(statsResults.posthoc_results).length > 0 && (
                              <div>
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Resultados Post-Hoc (Tukey/Nemenyi)</h5>
                                <div className="overflow-x-auto">
                                  <table className="w-full text-xs">
                                    <thead>
                                      <tr className="border-b border-slate-200 dark:border-slate-700">
                                        {["Comparación", "Diferencia", "Valor p", "Significativo"].map(h => (
                                          <th key={h} className="text-left py-2 px-2 text-slate-400 font-semibold">{h}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {Object.entries(statsResults.posthoc_results).map(([comparison, result]: any) => (
                                        <tr key={comparison} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                                          <td className="py-1.5 px-2 font-medium text-slate-700 dark:text-slate-300">{comparison}</td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof result.diff === 'number' ? result.diff.toFixed(4) : '—'}</td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">
                                            <span className={result.p_val < 0.05 ? "text-rose-600 dark:text-rose-400 font-bold" : ""}>
                                              {typeof result.p_val === 'number' ? result.p_val.toFixed(4) : '—'}
                                            </span>
                                          </td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">
                                            {result.significant ? "✅" : "❌"}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}

                            {/* 4. Pairwise Comparisons (T-Student/Wilcoxon) */}
                            {statsResults.pairwise_comparisons && Object.keys(statsResults.pairwise_comparisons).length > 0 && (
                              <div>
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Comparaciones Pareadas (Mejor vs Resto)</h5>
                                <div className="overflow-x-auto">
                                  <table className="w-full text-xs">
                                    <thead>
                                      <tr className="border-b border-slate-200 dark:border-slate-700">
                                        {["Comparación", "Test", "Estadístico", "Valor p", "Significativo"].map(h => (
                                          <th key={h} className="text-left py-2 px-2 text-slate-400 font-semibold">{h}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {Object.entries(statsResults.pairwise_comparisons).map(([comparison, result]: any) => (
                                        <tr key={comparison} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                                          <td className="py-1.5 px-2 font-medium text-slate-700 dark:text-slate-300">{comparison}</td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{result.test_name}</td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{typeof result.stat === 'number' ? result.stat.toFixed(4) : '—'}</td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">
                                            <span className={result.p_val < 0.05 ? "text-rose-600 dark:text-rose-400 font-bold" : ""}>
                                              {typeof result.p_val === 'number' ? result.p_val.toFixed(4) : '—'}
                                            </span>
                                          </td>
                                          <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">
                                            {result.significant ? "✅" : "❌"}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}

                            {/* 5. Wilcoxon Test */}
                            {statsResults.wilcoxon && (
                              <div>
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Test de Wilcoxon (Clásico vs Híbrido)</h5>
                                <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/20">
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div>
                                      <span className="text-xs font-semibold text-slate-500 block">Mejor Clásico</span>
                                      <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{statsResults.wilcoxon.best_classic}</span>
                                    </div>
                                    <div>
                                      <span className="text-xs font-semibold text-slate-500 block">Mejor Híbrido</span>
                                      <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{statsResults.wilcoxon.best_hybrid}</span>
                                    </div>
                                    <div>
                                      <span className="text-xs font-semibold text-slate-500 block">Estadístico</span>
                                      <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400">{statsResults.wilcoxon.stat?.toFixed(4) ?? "—"}</span>
                                    </div>
                                    <div>
                                      <span className="text-xs font-semibold text-slate-500 block">Valor p</span>
                                      <span className={`text-sm font-extrabold ${statsResults.wilcoxon.p_val < 0.05 ? "text-rose-600 dark:text-rose-400" : "text-slate-600 dark:text-slate-300"}`}>
                                        {statsResults.wilcoxon.p_val?.toFixed(4) ?? "—"}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* 6. Bootstrap CI section */}
                            {bootstrapCiData && (
                              <div>
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-4">Intervalos de Confianza Bootstrap</h5>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {Object.entries(bootstrapCiData).map(([name, ci]: any) => (
                                    <div key={name} className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/30">
                                      <span className="text-xs font-bold text-slate-400 block">{name}</span>
                                      <span className="text-lg font-extrabold text-emerald-600 dark:text-emerald-400 block">
                                        {`${(ci[0] * 100).toFixed(2)}% - ${(ci[1] * 100).toFixed(2)}%`}
                                      </span>
                                      <span className="text-xs text-slate-500">Intervalo de confianza basado en remuestreo bootstrap (n=1000)</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* 7. Interpretations */}
                            {(statsResults.interpretations || modelResults.stats_interpretation) && (
                              <div className="p-4 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50 dark:bg-slate-950/20">
                                <h5 className="font-bold text-sm text-slate-600 dark:text-slate-300 mb-3">Interpretación</h5>
                                <div className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-line">
                                  {statsResults.interpretations || modelResults.stats_interpretation}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Download reports row */}
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-4">
                    <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">{t("AutoML.report_section")}</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <button 
                        onClick={() => handleDownloadReport('pdf')}
                        className="p-4 border border-slate-200 dark:border-slate-800 hover:border-emerald-500 rounded-xl flex items-center justify-between text-xs font-semibold bg-slate-50/50 dark:bg-slate-950/20 cursor-pointer"
                      >
                        <span className="flex items-center gap-2">
                          <FileText className="text-red-500" />
                          {t("AutoML.download_pdf")}
                        </span>
                        <Download size={14} />
                      </button>
                      
                      <button 
                        onClick={() => handleDownloadReport('docx')}
                        className="p-4 border border-slate-200 dark:border-slate-800 hover:border-emerald-500 rounded-xl flex items-center justify-between text-xs font-semibold bg-slate-50/50 dark:bg-slate-950/20 cursor-pointer"
                      >
                        <span className="flex items-center gap-2">
                          <FileText className="text-blue-500" />
                          {t("AutoML.download_docx")}
                        </span>
                        <Download size={14} />
                      </button>
                      
                      <button 
                        onClick={() => handleDownloadReport('xlsx')}
                        className="p-4 border border-slate-200 dark:border-slate-800 hover:border-emerald-500 rounded-xl flex items-center justify-between text-xs font-semibold bg-slate-50/50 dark:bg-slate-950/20 cursor-pointer"
                      >
                        <span className="flex items-center gap-2">
                          <FileSpreadsheet className="text-emerald-500" />
                          {t("AutoML.download_xlsx")}
                        </span>
                        <Download size={14} />
                      </button>
                    </div>
                  </div>

                  {/* History Tab */}
                  {activeAutomlTab === "history" && (
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">📜 Historial de Auditoría MLOps</h4>
                          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Registro de ejecuciones del pipeline de AutoML y métricas históricas.</p>
                        </div>
                        <span className="text-xs font-semibold px-2.5 py-1 bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 rounded-lg">
                          {experimentHistory.length} Experimentos Registrados
                        </span>
                      </div>

                      {experimentHistory.length > 0 ? (
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-slate-200 dark:border-slate-700">
                                {["ID","Fecha","Modelo Ganador","Accuracy","F1-Score","Estrategia Tuning"].map(h => (
                                  <th key={h} className="text-left py-2.5 px-3 text-slate-400 font-semibold">{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {experimentHistory.map((exp: any) => (
                                <tr key={exp.id} className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                                  <td className="py-2.5 px-3 font-mono text-slate-500">#{exp.id}</td>
                                  <td className="py-2.5 px-3 text-slate-600 dark:text-slate-400">{exp.run_date}</td>
                                  <td className="py-2.5 px-3 font-semibold text-emerald-600 dark:text-emerald-400">{exp.best_model_name}</td>
                                  <td className="py-2.5 px-3 font-bold text-slate-800 dark:text-white">{(exp.accuracy * 100).toFixed(2)}%</td>
                                  <td className="py-2.5 px-3 text-slate-600 dark:text-slate-400">{exp.f1_score.toFixed(4)}</td>
                                  <td className="py-2.5 px-3 text-slate-500 uppercase text-[10px] font-bold tracking-wider">{exp.tuning_method}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-xs text-slate-500 text-center py-6">No hay ejecuciones anteriores registradas en el historial.</p>
                      )}
                    </div>
                  )}
                </div>
              )}
              
            </div>
          )}
          
          {/* TAB 4: PREDICCIÓN TABULAR EN VIVO */}
          {currentTab === "predict_tabular" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Formulario de entradas */}
              <div className="lg:col-span-1 p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl space-y-4">
                <div>
                  <h3 className="font-extrabold text-lg text-slate-800 dark:text-white flex items-center gap-2">
                    <Cpu className="text-emerald-500" />
                    Predicción Tabular en Vivo
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Ingrese variables agronómicas del cultivo para clasificar usando el mejor pipeline (.pkl).
                  </p>
                </div>

                <form onSubmit={handlePredictTabular} className="space-y-3">
                  {Object.keys(tabularFeatures).map((key) => (
                    <div key={key}>
                      <label className="text-xs font-semibold text-slate-600 dark:text-slate-400 block mb-1 capitalize">
                        {key.replace(/_/g, ' ')}
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={tabularFeatures[key]}
                        onChange={(e) => setTabularFeatures({ ...tabularFeatures, [key]: parseFloat(e.target.value) || 0 })}
                        className="w-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500"
                        required
                      />
                    </div>
                  ))}

                  <button
                    type="submit"
                    disabled={predictingTabular}
                    className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-2.5 rounded-xl text-sm transition-transform active:scale-[0.98] shadow-lg shadow-emerald-500/20 mt-4"
                  >
                    {predictingTabular ? "Clasificando..." : "Ejecutar Predicción"}
                  </button>
                </form>
              </div>

              {/* Tarjeta de resultados */}
              <div className="lg:col-span-2 space-y-6">
                {tabularPredictionResult ? (
                  <div className="p-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl space-y-6">
                    <div className="p-4 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/40 dark:to-teal-950/30 border border-emerald-300 dark:border-emerald-700/50 rounded-xl flex items-center justify-between flex-wrap gap-4 shadow-sm">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                          <span className="text-xl">🏆</span>
                        </div>
                        <div>
                          <p className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
                            Modelo Ejecutado
                          </p>
                          <p className="text-lg font-extrabold text-slate-800 dark:text-white">
                            {tabularPredictionResult.model_name}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <span className="block text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">
                            {(tabularPredictionResult.confidence * 100).toFixed(2)}%
                          </span>
                          <span className="text-xs text-slate-500 font-semibold">Nivel de Confianza</span>
                        </div>
                        <div className="text-right pl-4 border-l border-emerald-200 dark:border-emerald-800">
                          <span className="block text-lg font-extrabold text-slate-900 dark:text-white">
                            {tabularPredictionResult.prediction}
                          </span>
                          <span className="text-xs text-slate-500 font-semibold">Clase Predicha</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl text-sm leading-relaxed text-slate-700 dark:text-slate-300">
                      <strong>Interpretación Agronómica:</strong> {tabularPredictionResult.interpretation}
                    </div>

                    {tabularPredictionResult.probabilities && Object.keys(tabularPredictionResult.probabilities).length > 0 && (
                      <div className="space-y-3">
                        <h4 className="font-bold text-sm text-slate-400 uppercase tracking-wider">Probabilidades por Clase</h4>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                          {Object.entries(tabularPredictionResult.probabilities).map(([cls, prob]: any) => (
                            <div key={cls} className="p-3 border border-slate-200 dark:border-slate-800 rounded-xl bg-slate-50/50 dark:bg-slate-950/30 text-center">
                              <span className="text-xs text-slate-400 block font-semibold">{cls}</span>
                              <span className="text-base font-extrabold text-emerald-600 dark:text-emerald-400">
                                {(prob * 100).toFixed(2)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-12 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl text-center space-y-3">
                    <span className="text-4xl block">📊</span>
                    <h4 className="text-base font-bold text-slate-700 dark:text-slate-300">Esperando ejecución</h4>
                    <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
                      Complete los valores agronómicos a la izquierda y presione "Ejecutar Predicción" para consultar la respuesta del modelo en tiempo real.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 5: HISTORIAL MLOPS DEDICADO EN EL MENÚ PRINCIPAL */}
          {currentTab === "history" && (
            <div className="space-y-6 animate-in fade-in duration-300">
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h2 className="text-xl font-extrabold text-slate-800 dark:text-white flex items-center gap-2">
                    <History className="text-emerald-500" />
                    Historial de Experimentos y Auditoría MLOps
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Registro centralizado de todas las corridas de entrenamiento, modelos evaluados y métricas históricas.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold px-3 py-1.5 bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 rounded-xl border border-emerald-300 dark:border-emerald-700/50">
                    {experimentHistory.length} Experimentos Registrados
                  </span>
                </div>
              </div>

              {/* Tarjetas resumen del historial */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">Total Corridas</span>
                  <span className="text-3xl font-extrabold text-slate-800 dark:text-white">{experimentHistory.length}</span>
                </div>
                <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">Mejor Accuracy Registrado</span>
                  <span className="text-3xl font-extrabold text-emerald-600 dark:text-emerald-400">
                    {experimentHistory.length > 0
                      ? `${(Math.max(...experimentHistory.map((h: any) => h.accuracy || 0)) * 100).toFixed(2)}%`
                      : "—"}
                  </span>
                </div>
                <div className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">Último Modelo Ganador</span>
                  <span className="text-xl font-extrabold text-blue-600 dark:text-blue-400 truncate block">
                    {experimentHistory.length > 0 ? experimentHistory[0]?.best_model_name : "—"}
                  </span>
                </div>
              </div>

              {/* Tabla principal de auditoría MLOps */}
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6">
                {experimentHistory.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 dark:border-slate-700">
                          {["ID Run", "Fecha / Hora", "Modelo Ganador", "Accuracy", "F1-Score", "Estrategia Tuning"].map(h => (
                            <th key={h} className="text-left py-3 px-4 text-slate-400 font-bold uppercase tracking-wider">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {experimentHistory.map((exp: any) => (
                          <tr key={exp.id} className="border-b border-slate-100 dark:border-slate-800/60 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                            <td className="py-3 px-4 font-mono font-bold text-slate-500">#{exp.id}</td>
                            <td className="py-3 px-4 text-slate-600 dark:text-slate-350">{exp.run_date}</td>
                            <td className="py-3 px-4 font-bold text-emerald-600 dark:text-emerald-400">{exp.best_model_name}</td>
                            <td className="py-3 px-4 font-extrabold text-slate-900 dark:text-white">{(exp.accuracy * 100).toFixed(2)}%</td>
                            <td className="py-3 px-4 text-slate-600 dark:text-slate-300 font-mono">{exp.f1_score.toFixed(4)}</td>
                            <td className="py-3 px-4">
                              <span className="bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider border border-slate-200 dark:border-slate-700">
                                {exp.tuning_method}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="py-12 text-center space-y-2">
                    <span className="text-3xl block">📜</span>
                    <p className="text-sm font-bold text-slate-600 dark:text-slate-400">No se registran corridas en el historial</p>
                    <p className="text-xs text-slate-500">Ejecute el pipeline de AutoML para generar nuevos registros de entrenamiento.</p>
                  </div>
                )}
              </div>
            </div>
          )}
          
        </div>
      </main>
      
      {/* --- Floating Chatbot Bubble & Card --- */}
      <div className="fixed bottom-6 right-6 z-50">
        {chatOpen ? (
          <div className="w-80 h-[480px] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-bottom-5 duration-200">
            {/* Chat header */}
            <div className="bg-slate-900 text-white p-4 flex items-center justify-between">
              <span className="font-extrabold text-sm flex items-center gap-1.5">
                🤖 MaizIA
              </span>
              <div className="flex items-center gap-1">
                <button 
                  onClick={() => setSpeechEnabled(!speechEnabled)}
                  className="p-1 text-slate-400 hover:text-white rounded transition-colors"
                >
                  {speechEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
                </button>
                <button 
                  onClick={() => setChatHistory([])}
                  className="p-1 text-slate-400 hover:text-white rounded transition-colors"
                >
                  <Trash2 size={16} />
                </button>
                <button 
                  onClick={() => setChatOpen(false)}
                  className="p-1 text-slate-400 hover:text-white rounded transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
            </div>
            
            {/* Chat scrollable body */}
            <div className="flex-1 p-4 overflow-y-auto space-y-3 bg-slate-50/50 dark:bg-slate-950/20">
              {chatHistory.length === 0 ? (
                <div className="text-center py-10 space-y-3">
                  <div className="w-16 h-16 rounded-full border border-yellow-400/30 bg-yellow-500/10 flex items-center justify-center mx-auto text-yellow-500 shadow-lg">
                    <span className="text-2xl">🤖</span>
                  </div>
                  <h4 className="font-bold text-xs uppercase tracking-wider text-slate-400">{t("Chatbot.title")}</h4>
                  <p className="text-xs text-slate-500 leading-relaxed max-w-[200px] mx-auto">{t("Chatbot.welcome")}</p>
                </div>
              ) : (
                chatHistory.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`p-3 rounded-2xl text-xs max-w-[85%] leading-relaxed ${msg.role === "user" ? "bg-emerald-600 text-white rounded-br-none" : "bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-bl-none text-slate-700 dark:text-slate-350"}`}>
                      {msg.text}
                    </div>
                  </div>
                ))
              )}
              <div ref={chatBottomRef} />
            </div>
            
            {/* Chat input footer */}
            <form onSubmit={handleChatSubmit} className="p-3 border-t border-slate-200 dark:border-slate-800 flex items-center gap-2 bg-white dark:bg-slate-900">
              <button 
                type="button"
                onClick={toggleRecording}
                className={`p-2 rounded-full border transition-all ${isRecording ? "bg-red-500 text-white border-red-500 animate-pulse" : "bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-300"}`}
              >
                <Mic size={16} />
              </button>
              
              <input 
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={t("Chatbot.placeholder")}
                className="flex-1 bg-slate-100 dark:bg-slate-800 border-none rounded-full px-4 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 dark:text-white"
              />
              
              <button 
                type="submit"
                className="p-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-full active:scale-95 transition-transform"
              >
                <Send size={16} />
              </button>
            </form>
          </div>
        ) : (
          <button 
            onClick={() => setChatOpen(true)}
            className="w-14 h-14 bg-emerald-600 hover:bg-emerald-700 active:scale-95 transition-transform text-white rounded-full flex items-center justify-center shadow-xl shadow-emerald-600/20 border border-emerald-500/20"
          >
            <span className="text-xl">💬</span>
          </button>
        )}
      </div>
      
    </div>
  );
}
