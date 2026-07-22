import os

def main():
    prunes = {'.venv', '.git', 'SteamLibrary', 'Epic Games', 'Riot Games', 'WindowsApps', 'WpSystem', 'capcut videos', 'gta', 'XboxGames', 'AppData'}
    for r, d, files in os.walk('D:\\'):
        # prune in-place
        d[:] = [dirname for dirname in d if dirname not in prunes]
        for f in files:
            filepath = os.path.join(r, f)
            if any(k in f.lower() or k in r.lower() for k in ['taco', 'trashnet', 'cashcrow']):
                print(filepath)

if __name__ == '__main__':
    main()
