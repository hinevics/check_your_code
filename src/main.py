import logging

from telegram.ext import ApplicationBuilder, CommandHandler

from config import TG_TOKEN

# Настройка журналирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update, context):
    return context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Я бот Telegram.")


def main():
    app = ApplicationBuilder().token(TG_TOKEN).build()
    start_handler = CommandHandler('start', start)
    app.add_handler(start_handler)
    app.run_polling()

if __name__ == '__main__':
    main()
