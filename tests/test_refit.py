import hashlib
import unittest

from scripts.refit_frozen import text_hash_matches


class FrozenRefitTests(unittest.TestCase):
    def test_git_newline_conversion_is_allowed_but_content_changes_are_rejected(self):
        lf=b"header,value\n0013_01,True\n"
        crlf=lf.replace(b"\n",b"\r\n")
        for original in [lf,crlf]:
            digest=hashlib.sha256(original).hexdigest()
            self.assertTrue(text_hash_matches(lf,digest))
            self.assertTrue(text_hash_matches(crlf,digest))
            self.assertFalse(text_hash_matches(lf.replace(b"True",b"False"),digest))


if __name__=="__main__":unittest.main()
