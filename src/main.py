import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler,
    ConversationHandler, filters,
)

from myconfig import TG_TOKEN
from check_code import check


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


CODE1, CODE2 = range(2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Я бот Telegram. Отправь мне сой код!")


async def comparer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user.name
    await update.message.reply_text(text=f"{user} send me your first code.")
    # await context.bot.(
    #     chat_id=update.effective_chat.id,
    #     text=f"{user} send me your first code.")
    return CODE1


async def get_code1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('get_code1')
    user = update.message.from_user.name
    context.user_data['code1'] = update.message.text
    print(context.user_data['code1'])
    await update.message.reply_text(
        text=f"{user}, send me the second code.")

    return CODE2


async def get_code2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('get_code2')
    user = update.message.from_user.name
    context.user_data['code2'] = update.message.text
    score = check(code1=context.user_data['code1'], code2=context.user_data['code2'])
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"{user}, the two codes are similar to {score}")

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return ConversationHandler.END


def main():
    app = ApplicationBuilder().token(TG_TOKEN).build()
    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler('comparer', comparer)],
        states={
            CODE1: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_code1)],
            CODE2: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_code2)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conversation_handler)
    app.run_polling()


if __name__ == '__main__':
    main()
