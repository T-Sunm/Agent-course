from typing import Literal

import autogen
from typing_extensions import Annotated

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["qwen2-vl-2b-instruct"]
    }
)

llm_config = {
    "cache_seed": 43,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,  # in seconds
}

currency_bot = autogen.AssistantAgent(
    name="currency_bot",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE "
                   "when the task is done.",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=5,
    code_execution_config=False
)

CurrencySymbol = Literal["USD", "EUR"]


def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
  if base_currency == quote_currency:  # Nếu bạn đổi cùng một loại tiền
    return 1.0
  elif base_currency == "USD" and quote_currency == "EUR":  # Nếu đổi từ USD → EUR
    return 1 / 1.09
  elif base_currency == "EUR" and quote_currency == "USD":  # Nếu đổi từ EUR → USD
    return 1 / 1.1
  else:
    raise ValueError(
        f"Unknown currencies: {base_currency}, {quote_currency}")

@user_proxy.register_function()
@currency_bot.register_for_llm(description="currency exchange rate")
def currency_calculator(
        base_amount: Annotated[float, "Amount of currency in base_currency"],
        base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
        quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR"
) -> str:
  quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
  return f"{quote_amount} - {quote_currency}"


user_proxy.initiate_chat(
    currency_bot,
    message="How much is 1.234.56 USD in EUR? "
)
