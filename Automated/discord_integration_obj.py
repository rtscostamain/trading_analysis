import discord
import nest_asyncio


class AbstractSingleActionClient(discord.Client):
    async def do_work(self):
        raise NotImplementedError()

    async def on_ready(self):
        print('We have logged in as {0.user}'.format(self))
        await self.do_work()
        await self.logout()
        await self.close()
        
        
        
class SendBuyMessageClient(AbstractSingleActionClient):
    def __init__(self, channel_id, buy_info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_id = channel_id
        self.buy_info = buy_info

    async def do_work(self):
        await self.wait_until_ready()
        
        channel = self.get_channel(self.channel_id)
        
        if channel is None:
            raise Exception(
                f"Channel with id {self.channel_id} does not seem to exist!"
            )

        
        title = "Buy " + buy_info["ticker"]
        description = buy_info["date"] + " Time to make $$$$"
        
        embed=discord.Embed(title=title, description=description, color=discord.Color.blue())
        
        #embed.add_field(name="Field 1 Title", value="This is the value for field 1. This is NOT an inline field.", inline=False) 
        embed.add_field(name="Study", value=buy_info["study"], inline=True)
        embed.add_field(name="Stock price", value=buy_info["boughtPrice"], inline=True)
        embed.add_field(name="Qty", value=buy_info["qty"], inline=True)
        
        #embed.set_footer(text="This is the footer. It contains text at the bottom of the embed")    

        result = await channel.send(embed=embed)
              
        
buy_info = {
    "ticker": "STNE",
    "date": "2021-11-78 11:50",
    "study": "RSI",
    "boughtPrice": 20.55,
    "qty": 1
    }

channel_id = 814324272436609027
token = "xxx"        
nest_asyncio.apply()
client = SendBuyMessageClient(channel_id, buy_info)
client.run(token)        