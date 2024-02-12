import cmd


class DatasetTools(cmd.Cmd):
    prompt = '/> '
    intro = 'This tool will help you generate or modify a dataset for SentiSense'

    def do_modify(self, line):
        """Modify an existing dataset"""
        print("Modifying a dataset\n")

    def do_create(self, line):
        """Create a new dataset"""
        print("Creating a new dataset\n")
    
    def do_quit(self, line):
        """Exit the CLI."""
        return True

if __name__ == '__main__':
    DatasetTools().cmdloop()