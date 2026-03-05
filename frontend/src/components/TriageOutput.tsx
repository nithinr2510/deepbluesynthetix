import { Copy, FileText, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import type { TriageResult } from "@/types/ticket";

interface TriageOutputProps {
    result: TriageResult;
}

const urgencyColors: Record<string, string> = {
    High: "bg-red-500/15 text-red-400 border-red-500/30",
    Critical: "bg-red-500/15 text-red-400 border-red-500/30",
    Medium: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30",
    Low: "bg-green-500/15 text-green-400 border-green-500/30",
};

const categoryColors: Record<string, string> = {
    Incident: "bg-orange-500/15 text-orange-400 border-orange-500/30",
    Problem: "bg-red-500/15 text-red-400 border-red-500/30",
    Request: "bg-blue-500/15 text-blue-400 border-blue-500/30",
    Change: "bg-purple-500/15 text-purple-400 border-purple-500/30",
};

export const TriageOutput = ({ result }: TriageOutputProps) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(result.draft_reply);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const urgencyClass = urgencyColors[result.urgency] || "bg-gray-500/15 text-gray-400 border-gray-500/30";
    const categoryClass = categoryColors[result.category] || "bg-gray-500/15 text-gray-400 border-gray-500/30";

    return (
        <div className="card-elevated rounded-xl p-6 space-y-5">
            <h2 className="text-lg font-semibold text-foreground">AI Triage Output</h2>

            {/* Category & Urgency Badges */}
            <div className="flex flex-wrap gap-3">
                <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${categoryClass}`}>
                    Category: {result.category}
                </div>
                <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${urgencyClass}`}>
                    Urgency: {result.urgency}
                </div>
            </div>

            {/* Confidence Bar */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-medium text-muted-foreground">Model Confidence</span>
                    <span className="text-xs font-mono text-primary">{result.confidence}%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-secondary overflow-hidden">
                    <div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-primary/60 transition-all duration-1000 ease-out"
                        style={{ width: `${Math.min(result.confidence, 100)}%` }}
                    />
                </div>
            </div>

            {/* Summary */}
            <div>
                <h3 className="text-sm font-semibold text-muted-foreground mb-1">AI Summary</h3>
                <p className="text-sm text-foreground font-medium leading-relaxed">{result.summary}</p>
            </div>

            {/* Draft Reply */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <h3 className="text-sm font-semibold text-muted-foreground">Draft Reply</h3>
                    <button
                        onClick={handleCopy}
                        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
                    >
                        {copied ? (
                            <>
                                <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                                Copied!
                            </>
                        ) : (
                            <>
                                <Copy className="w-3.5 h-3.5" />
                                Copy
                            </>
                        )}
                    </button>
                </div>
                <div className="rounded-lg border border-border bg-secondary/30 p-4 text-sm text-foreground leading-relaxed whitespace-pre-wrap font-mono">
                    {result.draft_reply}
                </div>
            </div>

            {/* Sources */}
            {result.sources && result.sources.length > 0 && (
                <div>
                    <h3 className="text-sm font-semibold text-muted-foreground mb-2">KB Citations</h3>
                    <div className="flex flex-wrap gap-2">
                        {result.sources.map((source, index) => (
                            <span
                                key={index}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono bg-secondary border border-border text-muted-foreground"
                            >
                                <FileText className="w-3 h-3" />
                                {source}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
