import json
import sys
import click

@click.command()
@click.argument('feat1', required=True)
@click.argument('feat2', required=True)
def cli(feat1, feat2):
	with open(feat1) as f:
		db1 = json.load(f)
	with open(feat2) as f:
		db2 = json.load(f)

	c = {}
	for k, v in db1.iteritems():
		c[k] = db1[k] + db2[k]

	json.dump(c, sys.stdout)

if __name__ == '__main__':
	cli()
