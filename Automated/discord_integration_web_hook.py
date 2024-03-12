from discord_webhook import DiscordWebhook, DiscordEmbed

webhook = DiscordWebhook(url='https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDOT9UQVJWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-cXQ7uIm', content='Webhook Message')
response = webhook.execute()

webhook = DiscordWebhook(url='https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDOT9UQVJWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXT7uIm', username="New Webhook Username")

embed = DiscordEmbed(title='Embed Title', description='Your Embed Description', color=242424)
embed.set_author(name='Author Name', url='https://github.com/lovvskillz', icon_url='https://avatars0.githubusercontent.com/u/14542790')
embed.set_footer(text='Embed Footer Text')
embed.set_timestamp()
embed.add_embed_field(name='Field 1', value='Lorem ipsum')
embed.add_embed_field(name='Field 2', value='dolor sit')
embed.add_embed_field(name='Field 3', value='amet consetetur')
embed.add_embed_field(name='Field 4', value='sadipscing elitr')

webhook.add_embed(embed)
response = webhook.execute()