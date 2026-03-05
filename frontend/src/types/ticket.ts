export interface TriageResult {
    category: string;
    urgency: string;
    summary: string;
    draft_reply: string;
    sources: string[];
    confidence: number;
}
