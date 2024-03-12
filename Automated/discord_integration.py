import discord
import os
import nest_asyncio

message = "Minha mensagem de teste"


client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    
    

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')


async def do_work(channel_msg):
    await client.wait_until_ready()
    channel = client.get_channel(814324272436609027)
    
    embed=discord.Embed(title="Sample Embed", url="https://realdrewdata.medium.com/", description="This is an embed that will show how to build an embed and the different components", color=discord.Color.blue())
    
    embed.add_field(name="Field 1 Title", value="This is the value for field 1. This is NOT an inline field.", inline=False) 
    embed.add_field(name="Field 2 Title", value="It is inline with Field 3", inline=True)
    embed.add_field(name="Field 3 Title", value="It is inline with Field 2", inline=True)
    
    embed.set_footer(text="This is the footer. It contains text at the bottom of the embed")    
    
    await channel.send(embed=embed)
    
    #channel = client.get_channel(814324272436609027)
    #await channel.send(channel_msg)
    #await client.close()


nest_asyncio.apply()

client.loop.create_task(do_work(message))
client.run("xxx")
client.logout()


#id channel = 814324272436609027
#https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDOT9UQVJWYZaHAuTJOay6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXQ7uIm
#{"type": 1, "id": "814330374658326538", "name": "Captain Hook", "avatar": null, "channel_id": "814324272436609027", "guild_id": "814324272436609024", "application_id": null, "token": "O6xsz2BWTCDOT9UQVJWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXQ7uIm"}