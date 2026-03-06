import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TicketForm } from "@/components/TicketForm";
import { TriageOutput } from "@/components/TriageOutput";
import { Zap, ShieldCheck } from "lucide-react";
import type { TriageResult } from "@/types/ticket";
import { toast } from "sonner";

const Index = () => {
    const [result, setResult] = useState<TriageResult | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (payload: {
        subject?: string;
        description: string;
        channel?: string;
        timestamp?: string;
    }) => {
        setLoading(true);
        setResult(null);
        try {
            const res = await fetch("http://localhost:8000/process-ticket", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error(`Server responded with ${res.status}`);
            const data: TriageResult = await res.json();
            setResult(data);
            toast.success("Ticket processed successfully!");
        } catch (err: any) {
            toast.error("Failed to reach HelpDeskAi", {
                description: err.message || "Ensure the backend is running at localhost:8000",
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background relative overflow-hidden">
            {/* Ambient glow */}
            <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-primary/5 rounded-full blur-[120px] pointer-events-none" />
            <div className="fixed bottom-0 right-0 w-[500px] h-[300px] bg-primary/3 rounded-full blur-[100px] pointer-events-none" />

            {/* Header */}
            <header className="relative z-10 border-b border-border/50 glass">
                <div className="container mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-primary/15 flex items-center justify-center glow-primary-sm">
                            <Zap className="w-5 h-5 text-primary" />
                        </div>
                        <h1 className="text-xl font-bold tracking-tight">
                            <span className="gradient-text">HelpDeskAi</span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <ShieldCheck className="w-4 h-4 text-success" />
                        <span className="font-mono text-xs">AI Engine Active</span>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 container mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                    {/* Left: Input */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <TicketForm onSubmit={handleSubmit} loading={loading} />
                    </motion.div>

                    {/* Right: Output */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                    >
                        <AnimatePresence mode="wait">
                            {loading ? (
                                <motion.div
                                    key="loader"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="card-elevated rounded-xl p-12 flex flex-col items-center justify-center min-h-[400px]"
                                >
                                    <div className="w-14 h-14 rounded-full border-2 border-primary/30 border-t-primary animate-spin-slow" />
                                    <p className="mt-6 text-sm text-muted-foreground font-mono">Processing with HelpDeskAi…</p>
                                </motion.div>
                            ) : result ? (
                                <motion.div
                                    key="result"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                >
                                    <TriageOutput result={result} />
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="card-elevated rounded-xl p-12 flex flex-col items-center justify-center min-h-[400px] text-center"
                                >
                                    <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center mb-4">
                                        <Zap className="w-7 h-7 text-muted-foreground" />
                                    </div>
                                    <p className="text-muted-foreground text-sm">Submit a ticket to see the AI triage output</p>
                                    <p className="text-muted-foreground/50 text-xs mt-1 font-mono">Powered by DistilBERT + Llama 3</p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                </div>
            </main>
        </div>
    );
};

export default Index;
