import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler,
    filters,
)

from src.myconfig import TG_TOKEN
from check_code import check


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Я бот Telegram. Отправь мне сой код!")


# def checking(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     code = ' '.join(context.args)
#     index = check(code)
#     return context.bot.send_message(
#         chat_id=update.effective_chat.id, text=f'score: {index}'
#     )


def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = update.message.text
    index = check(code)
    return context.bot.send_message(
        chat_id=update.effective_chat.id, text=f'score: {index}')


def main():
    app = ApplicationBuilder().token(TG_TOKEN).build()
    start_handler = CommandHandler('start', start)
    echo_handler = MessageHandler(filters.TEXT, echo)
    # checking_handler = CommandHandler('checking', checking)
    app.add_handler(start_handler)
    app.add_handler(echo_handler)
    # app.add_handler(checking_handler)
    app.run_polling()


if __name__ == '__main__':
    main()
