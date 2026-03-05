import { useState } from "react";
import { Send, Loader2 } from "lucide-react";

interface TicketFormProps {
    onSubmit: (payload: {
        subject?: string;
        description: string;
        channel?: string;
        timestamp?: string;
    }) => void;
    loading: boolean;
}

export const TicketForm = ({ onSubmit, loading }: TicketFormProps) => {
    const [subject, setSubject] = useState("");
    const [description, setDescription] = useState("");
    const [channel, setChannel] = useState("Email");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!description.trim()) return;
        onSubmit({
            subject,
            description,
            channel,
            timestamp: new Date().toISOString(),
        });
    };

    return (
        <div className="card-elevated rounded-xl p-6">
            <div className="mb-6">
                <h2 className="text-lg font-semibold text-foreground">Submit a Ticket</h2>
                <p className="text-sm text-muted-foreground mt-1">
                    Enter the support ticket details below for AI triage.
                </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                {/* Subject */}
                <div>
                    <label htmlFor="subject" className="block text-sm font-medium text-muted-foreground mb-1.5">
                        Subject
                    </label>
                    <input
                        id="subject"
                        type="text"
                        value={subject}
                        onChange={(e) => setSubject(e.target.value)}
                        placeholder="e.g. Cannot reset my password"
                        className="w-full rounded-lg border border-border bg-secondary/50 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
                    />
                </div>

                {/* Channel */}
                <div>
                    <label htmlFor="channel" className="block text-sm font-medium text-muted-foreground mb-1.5">
                        Channel
                    </label>
                    <select
                        id="channel"
                        value={channel}
                        onChange={(e) => setChannel(e.target.value)}
                        className="w-full rounded-lg border border-border bg-secondary/50 px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
                    >
                        <option value="Email">Email</option>
                        <option value="Chat">Chat</option>
                        <option value="Social Media">Social Media</option>
                        <option value="Phone">Phone</option>
                    </select>
                </div>

                {/* Description */}
                <div>
                    <label htmlFor="description" className="block text-sm font-medium text-muted-foreground mb-1.5">
                        Description <span className="text-destructive">*</span>
                    </label>
                    <textarea
                        id="description"
                        rows={6}
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="Describe the customer's issue in detail..."
                        required
                        className="w-full rounded-lg border border-border bg-secondary/50 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all resize-none"
                    />
                </div>

                {/* Submit */}
                <button
                    type="submit"
                    disabled={loading || !description.trim()}
                    className="w-full flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all glow-primary-sm"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Processing…
                        </>
                    ) : (
                        <>
                            <Send className="w-4 h-4" />
                            Process with Deep Blue AI
                        </>
                    )}
                </button>
            </form>
        </div>
    );
};
