class AIInsightsGenerator:
    @staticmethod
    def generate_insights(pipeline_results):
        insights = []
        metrics = pipeline_results['metrics']
        preds = pipeline_results['predictions']
        hist = pipeline_results['historical']
        feat_imp = pipeline_results['feature_importance']
        
        # Trend Analysis
        recent_avg = sum(x['Sales'] for x in hist[-7:]) / min(7, len(hist))
        future_avg = sum(x['Predicted_Sales'] for x in preds[:7]) / 7
        
        change_pct = ((future_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        if change_pct > 5:
            insights.append({
                "type": "positive",
                "title": "Upward Trend Detected",
                "message": f"Sales are predicted to increase by {change_pct:.1f}% over the next week. Consider increasing stock for high-performing products to avoid sellouts."
            })
        elif change_pct < -5:
            insights.append({
                "type": "warning",
                "title": "Sales Drop Predicted",
                "message": f"We forecast a {abs(change_pct):.1f}% drop in sales next week. It might be a good time to run promotional campaigns or flash sales."
            })
        else:
            insights.append({
                "type": "neutral",
                "title": "Stable Demand",
                "message": "Sales are expected to remain stable over the next week. Keep maintaining current inventory levels."
            })

        # Feature Importance Insights
        if feat_imp:
            top_feature = max(feat_imp, key=feat_imp.get)
            if top_feature == 'is_weekend' and feat_imp[top_feature] > 0.1:
                insights.append({
                    "type": "info",
                    "title": "Weekend Seasonality",
                    "message": "Weekends heavily influence your sales. Ensure adequate staffing and inventory levels as Friday approaches."
                })
            elif top_feature == 'lag_7':
                insights.append({
                    "type": "info",
                    "title": "Weekly Cyclic Patterns",
                    "message": "Your sales strongly repeat week-by-week. What sells well this Tuesday is highly likely to reproduce next Tuesday."
                })

        return insights

import os
import google.generativeai as genai

class ChatAssistant:
    def __init__(self):
        self.context = ""
        self.model = None
        self.chat_session = None
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.chat_session = self.model.start_chat(history=[])
    
    def update_context(self, pipeline_results, insights):
        model = pipeline_results['best_model']
        r2 = pipeline_results['metrics']['R2']
        
        self.context = f"The current best model is {model} with an R2 score of {r2:.2f}. "
        for ins in insights:
            self.context += f"{ins['title']}: {ins['message']} "
            
        if self.chat_session:
            sys_msg = (
                "You are an AI Sales Assistant for an SME. "
                "Here are the latest insights from the sales dataset: " + self.context +
                " Answer the user's questions based on this data. Be helpful, concise, and professional."
            )
            # Restart chat with the new context as the first invisible prompt
            self.chat_session = self.model.start_chat(history=[])
            try:
                self.chat_session.send_message(sys_msg)
            except Exception as e:
                print(f"Error initializing Gemini context: {e}")

    def reply(self, message):
        if not self.model:
            return "Please set the GEMINI_API_KEY in the .env file and restart the server to use the real AI."
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error communicating with Gemini: {str(e)}"
